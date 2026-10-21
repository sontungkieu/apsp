"""Publish committed main to GitHub, then atomically distribute its Git bundle on HF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request

BRANCH = 'main'
GITHUB = 'https://github.com/sontungkieu/apsp.git'
HF_REPO = 'codemaivanngu/apsp-lite-b200-cu128'


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--allow-untracked', action='store_true',
                        help='Preserve unrelated untracked files; publish committed history only')
    args = parser.parse_args()
    root = Path(git(Path(__file__).parent, 'rev-parse', '--show-toplevel'))
    if git(root, 'branch', '--show-current') != BRANCH:
        raise ValueError('Publish only canonical main')
    if git(root, 'remote', 'get-url', '--push', 'origin') != GITHUB:
        raise ValueError('Unexpected GitHub origin')
    status_args=['status','--porcelain'] + (['--untracked-files=no'] if args.allow_untracked else [])
    if git(root, *status_args):
        raise ValueError('Commit source edits/untracked files first')
    if git(root, 'rev-parse', '--is-shallow-repository') != 'false':
        raise ValueError('Full source history required')
    head = git(root, 'rev-parse', 'HEAD')
    if not args.publish:
        print('CHECK_ONLY ' + head)
        return
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    if api.whoami()['name'] != 'codemaivanngu':
        raise ValueError('Unexpected HF account')
    info = api.repo_info(HF_REPO, repo_type='dataset')
    if info.private:
        raise ValueError('Expected existing public distribution repo')
    parent = info.sha
    artifact = root / 'remote_artifacts'
    artifact.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='source-' + head[:8] + '-', dir=artifact))
    bundle = stage / 'apsp.bundle'
    git(root, 'bundle', 'create', str(bundle), 'refs/heads/' + BRANCH)
    git(root, 'bundle', 'verify', str(bundle))
    if git(root, 'bundle', 'list-heads', str(bundle)) != head + ' refs/heads/' + BRANCH:
        raise ValueError('Bundle ref mismatch')
    def verify_github():
        if git(root, 'ls-remote', 'origin', 'refs/heads/' + BRANCH) != head + '\trefs/heads/' + BRANCH:
            raise ValueError('GitHub ref differs')
    git(root, 'push', '--set-upstream', 'origin', 'HEAD:refs/heads/' + BRANCH)
    verify_github()
    manifest = {'format': 'git-bundle', 'history': 'complete', 'github': GITHUB,
                'source_branch': BRANCH, 'source_commit': head, 'bundle_file': bundle.name,
                'bundle_bytes': bundle.stat().st_size, 'bundle_sha256': sha(bundle),
                'hf_repo': HF_REPO, 'updater_files': {}}
    helpers = [root/'env/lite/update_source.py', root/'env/lite/update-apsp.sh']
    for helper in helpers:
        manifest['updater_files'][helper.name] = {'bytes': helper.stat().st_size, 'sha256': sha(helper)}
    mp = stage / 'source-manifest.json'
    mp.write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    # A separate source channel avoids touching/re-uploading the 50 GB payload release.
    files = [bundle, mp, *helpers]
    commit = api.create_commit(repo_id=HF_REPO, repo_type='dataset', parent_commit=parent,
        operations=[CommitOperationAdd(path_in_repo='source/'+p.name, path_or_fileobj=str(p)) for p in files],
        commit_message='Sync GitHub APSP main ' + head)
    for local in files:
        url = f'https://huggingface.co/datasets/{HF_REPO}/resolve/{commit.oid}/source/{local.name}'
        with urllib.request.urlopen(url, timeout=120) as response:
            body = response.read(local.stat().st_size + 1)
        if len(body) != local.stat().st_size or hashlib.sha256(body).hexdigest() != sha(local):
            raise ValueError('Anonymous HF readback mismatch: ' + local.name)
    verify_github()
    receipt = {'pass': True, 'source_commit': head, 'source_branch': BRANCH, 'github': GITHUB,
               'hf_repo': HF_REPO, 'hf_revision': commit.oid, 'manifest_sha256': sha(mp),
               'bundle_sha256': sha(bundle), 'github_verified': True, 'anonymous_hf_verified': True}
    (stage/'receipt.json').write_bytes((json.dumps(receipt, indent=2)+'\n').encode())
    print('DUAL_PUBLISH_PASS ' + json.dumps(receipt))
    print('RECEIPT=' + str(stage/'receipt.json'))


if __name__ == '__main__':
    main()
