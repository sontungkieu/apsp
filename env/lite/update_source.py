"""Update a dedicated server checkout from the public HF source channel, fast-forward only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

REPO = 'codemaivanngu/apsp-lite-b200-cu128'
GITHUB = 'https://github.com/sontungkieu/apsp.git'


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def download(url, target, limit):
    proxy = os.environ.get('PROXY') or os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
    if not proxy and os.environ.get('APSP_ALLOW_DIRECT') != '1':
        raise ValueError('Proxy env missing; follow PROXY_DOWNLOAD_NOTE.md')
    cmd = ['curl', '--silent', '--fail', '--location', '--retry', '3', '--connect-timeout', '20',
           '--max-time', '600', '--max-filesize', str(limit)]
    cmd += ['--proxy', proxy, '--noproxy', ''] if proxy else ['--noproxy', '*']
    result = subprocess.run([*cmd, url, '-o', str(target)], stderr=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError('Source download failed: curl rc=' + str(result.returncode))


def verify_checkout(target):
    if target.is_symlink() or Path(git(target, 'rev-parse', '--show-toplevel')).resolve() != target.resolve():
        raise ValueError('Expected dedicated non-symlink source checkout')
    if git(target, 'branch', '--show-current') != 'main' or git(target, 'status', '--porcelain'):
        raise ValueError('Dirty or non-main checkout preserved; stop')


def apply_bundle(bundle, manifest, target):
    """No resets: preserve dirty/diverged state and reject unexpected bundle refs."""
    head = manifest['source_commit']
    if manifest['source_branch'] != 'main' or not re.fullmatch('[0-9a-f]{40}', head):
        raise ValueError('Unexpected source identity')
    if manifest['github'] != GITHUB or manifest['bundle_file'] != 'apsp.bundle':
        raise ValueError('Unexpected distribution identity')
    if bundle.stat().st_size != manifest['bundle_bytes'] or hashlib.sha256(bundle.read_bytes()).hexdigest() != manifest['bundle_sha256']:
        raise ValueError('Bundle hash/size mismatch')
    if git(target.parent, 'bundle', 'list-heads', str(bundle)) != head + ' refs/heads/main':
        raise ValueError('Bundle ref mismatch')
    if target.exists():
        verify_checkout(target)
        git(target, 'bundle', 'verify', str(bundle))
        git(target, 'fetch', '--no-tags', str(bundle), 'refs/heads/main')
        if git(target, 'rev-parse', 'FETCH_HEAD') != head:
            raise ValueError('Fetched source mismatch')
        if subprocess.run(['git', 'merge-base', '--is-ancestor', 'HEAD', head], cwd=target).returncode:
            raise ValueError('Diverged or older source; checkout preserved')
        git(target, 'merge', '--ff-only', head)
    else:
        # Clone beside destination, then publish atomically; keep failed staging for diagnosis.
        stage = Path(tempfile.mkdtemp(prefix='.'+target.name+'-clone-', dir=target.parent))/'checkout'
        git(target.parent, 'clone', '--branch', 'main', str(bundle), str(stage))
        git(stage, 'remote', 'set-url', 'origin', GITHUB)
        if git(stage, 'rev-parse', 'HEAD') != head:
            raise ValueError('Clone source mismatch')
        stage.rename(target)
        stage.parent.rmdir()
    verify_checkout(target)
    if git(target, 'rev-parse', 'HEAD') != head:
        raise ValueError('Final source mismatch')
    return head


def main():
    import fcntl  # server runs Linux; no dependency installation needed
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=Path('/workspace/storage-shared/nlp/tungks/APSP-git'))
    parser.add_argument('--revision', help='Immutable HF commit; omitted resolves main once, then pins all reads')
    args = parser.parse_args()
    target = args.target.absolute()
    if target.is_symlink() or target.name in ('', '.', '..') or target == Path(target.anchor):
        raise ValueError('Unsafe checkout destination')
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent/('.'+target.name+'.source-update.lock')).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            verify_checkout(target)
        work = Path(tempfile.mkdtemp(prefix='apsp-source-', dir=target.parent))
        revision = args.revision
        if revision is None:
            download(f'https://huggingface.co/api/datasets/{REPO}/revision/main', work/'head.json', 2<<20)
            revision = json.loads((work/'head.json').read_text())['sha']
        if not re.fullmatch('[0-9a-f]{40}', revision):
            raise ValueError('HF revision must be an immutable commit')
        base = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/source/'
        download(base+'source-manifest.json', work/'source-manifest.json', 2<<20)
        manifest = json.loads((work/'source-manifest.json').read_text())
        if not isinstance(manifest['bundle_bytes'], int) or not 0 < manifest['bundle_bytes'] <= 128<<20:
            raise ValueError('Source bundle exceeds download bound')
        download(base+'apsp.bundle', work/'apsp.bundle', 128<<20)
        head = apply_bundle(work/'apsp.bundle', manifest, target)
        receipt = {'pass': True, 'source_commit': head, 'hf_revision': revision,
                   'target': str(target), 'bundle_sha256': manifest['bundle_sha256']}
        (work/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print('UPDATE_PASS ' + json.dumps(receipt))


if __name__ == '__main__':
    main()
