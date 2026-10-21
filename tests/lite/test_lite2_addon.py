"""Bounded stdlib checks for delta restore, legacy cache preservation and unsafe archives."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('lite2_addon',Path(__file__).resolve().parents[2]/'env/lite/lite2_addon.py')
addon=importlib.util.module_from_spec(spec);spec.loader.exec_module(addon)


class AddonTest(unittest.TestCase):
    def make_archive(self,root,bad=False):
        old=root/'companion';p=old/'hf/hub/models--legacy/snapshots/abc/weights';p.parent.mkdir(parents=True);p.write_bytes(b'keep')
        name='hf/hub/models--new/snapshots/def/weights';body=b'new'
        manifest={'requires_companion':{p.relative_to(old).as_posix():{'bytes':4,'sha256':addon.sha(p)}},
                  'files':{name:{'bytes':len(body),'sha256':'0'*64 if bad else hashlib.sha256(body).hexdigest()}}}
        archive=root/'addon.tar'
        with tarfile.open(archive,'w') as tf:
            for n,b in [(name,body),('LITE2_ADDON_MANIFEST.json',json.dumps(manifest).encode())]:
                info=tarfile.TarInfo(n);info.size=len(b);tf.addfile(info,io.BytesIO(b))
        return archive,old

    def test_hash_and_member_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            archive,_=self.make_archive(Path(d),bad=True)
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):addon.verify(archive,'0'*64)
            with self.assertRaisesRegex(ValueError,'Member hash'):addon.verify(archive,addon.sha(archive))

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'bad.tar'
            with tarfile.open(archive,'w') as tf:
                info=tarfile.TarInfo('../outside');info.size=1;tf.addfile(info,io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError,'Unsafe'):addon.verify(archive,addon.sha(archive))

    @unittest.skipIf(os.name=='nt','Linux cache-view symlinks; run separately in WSL')
    def test_restore_reuse_idempotence_preserves_old(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive,old=self.make_archive(root);target=root/'restored'
            self.assertEqual(addon.restore(archive,target,old,addon.sha(archive))['status'],'restored')
            self.assertEqual((target/'hf/hub/models--legacy/snapshots/abc/weights').read_bytes(),b'keep')
            self.assertEqual((old/'hf/hub/models--legacy/snapshots/abc/weights').read_bytes(),b'keep')
            self.assertFalse((old/'hf/hub/models--new').exists())
            self.assertEqual(addon.restore(archive,target,old,addon.sha(archive))['status'],'already_restored')
            (root/'existing').mkdir()
            with self.assertRaises(FileExistsError):addon.restore(archive,root/'existing',old,addon.sha(archive))


if __name__=='__main__':unittest.main()
