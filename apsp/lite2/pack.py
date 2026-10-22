"""Streaming tar parts and checked restore; never joins a second huge tar on disk."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from apsp.lite.storage import atomic_json
from .contracts import file_hash
from .export import validate_export


class PartWriter:
    def __init__(self, out, limit):
        self.out, self.limit, self.size, self.file, self.hasher, self.parts = Path(out), limit, 0, None, None, []

    def write(self, data):
        length = len(data)
        while data:
            if self.file is None:
                name = f'data.tar.part{len(self.parts):04d}'
                self.file = (self.out / name).open('xb')
                self.hasher, self.size = hashlib.sha256(), 0
                self.name = name
            chunk = data[:self.limit - self.size]
            self.file.write(chunk); self.hasher.update(chunk); self.size += len(chunk)
            data = data[len(chunk):]
            if self.size == self.limit:
                self.finish()
        return length

    def finish(self):
        if self.file is not None:
            self.file.flush(); self.file.close()
            self.parts.append({'file': self.name, 'bytes': self.size, 'sha256': self.hasher.hexdigest()})
            self.file = None

    def flush(self):
        if self.file is not None:
            self.file.flush()


class PartReader:
    def __init__(self, root, parts):
        self.paths = iter(Path(root) / p['file'] for p in parts)
        self.file = None

    def read(self, size):
        result = bytearray()
        while len(result) < size:
            if self.file is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.file = path.open('rb')
            data = self.file.read(size - len(result))
            if data:
                result.extend(data)
            else:
                self.file.close(); self.file = None
        return bytes(result)

    def close(self):
        if self.file is not None:
            self.file.close()


def pack(release, out, part_bytes=1_000_000_000):
    release, out = Path(release), Path(out)
    validate_export(release)
    if part_bytes < 1024 or part_bytes > 2_000_000_000:
        raise ValueError('Part size must be between 1KiB (tests) and 2GB')
    out.mkdir(parents=True, exist_ok=False)
    writer = PartWriter(out, part_bytes)
    try:
        with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(release.rglob('*')):
                if path.is_symlink():
                    raise ValueError('Export must contain materialized files, not symlinks')
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(release).as_posix(), recursive=False)
    finally:
        writer.finish()
    total = sum(p['bytes'] for p in writer.parts)
    if total + 1_000_000 > 80_000_000_000:
        raise ValueError('Tar parts plus metadata exceed 80GB; preserved for diagnosis')
    report = {'format': 'streaming-tar-parts/v1', 'total_archive_bytes': total, 'parts': writer.parts,
              'release_manifest_sha256': file_hash(release / 'RELEASE_MANIFEST.json')}
    atomic_json(out / 'PARTS.json', report)
    return report


def restore(parts_dir, out):
    parts_dir, out = Path(parts_dir), Path(out)
    manifest = json.loads((parts_dir / 'PARTS.json').read_text())
    if manifest['format'] != 'streaming-tar-parts/v1' or not manifest['parts']:
        raise ValueError('Invalid parts manifest')
    total = 0
    seen = set()
    for part in manifest['parts']:
        path = Path(part['file'])
        if path.name != part['file'] or part['file'] in seen:
            raise ValueError('Unsafe/duplicate part name')
        seen.add(part['file'])
        if (parts_dir / path).stat().st_size != part['bytes'] or file_hash(parts_dir / path) != part['sha256']:
            raise ValueError('Part checksum/size differs')
        total += part['bytes']
    if total != manifest['total_archive_bytes'] or total > 80_000_000_000:
        raise ValueError('Archive size differs')
    out.mkdir(parents=True, exist_ok=False)
    reader = PartReader(parts_dir, manifest['parts'])
    extracted = set()
    try:
        with tarfile.open(fileobj=reader, mode='r|') as archive:
            for member in archive:
                name = Path(member.name)
                if (not member.isfile() or name.is_absolute() or '..' in name.parts or
                        member.name in extracted or not (out / name).resolve().is_relative_to(out.resolve())):
                    raise ValueError('Unsafe/duplicate archive member')
                extracted.add(member.name)
                target = out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as dest:
                    import shutil
                    shutil.copyfileobj(source, dest, 1 << 20)
    finally:
        reader.close()
    if file_hash(out / 'RELEASE_MANIFEST.json') != manifest['release_manifest_sha256']:
        raise ValueError('Restored release manifest differs')
    return validate_export(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('pack'); a.add_argument('--release', required=True); a.add_argument('--out', required=True)
    a.add_argument('--part-bytes', type=int, default=1_000_000_000)
    b = sub.add_parser('restore'); b.add_argument('--parts', required=True); b.add_argument('--out', required=True)
    args = p.parse_args()
    print(json.dumps(pack(args.release, args.out, args.part_bytes) if args.mode == 'pack' else restore(args.parts, args.out)))
