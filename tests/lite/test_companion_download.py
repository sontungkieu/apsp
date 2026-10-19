import hashlib
import importlib.util
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import pytest

spec=importlib.util.spec_from_file_location('download_companion',Path(__file__).resolve().parents[2]/'env/lite/download_companion.py')
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


def test_twelve_workers_resume_and_private_urls_not_saved(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'BLOCK',1024)
    data=[bytes([i])*1024 for i in range(12)]
    full=b''.join(data); barrier=threading.Barrier(12)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            i=int(self.path.split('?')[0][1:])
            barrier.wait(timeout=10)
            self.send_response(200); self.end_headers(); self.wfile.write(data[i])
        def log_message(self,*args): pass
    class Server(ThreadingHTTPServer): request_queue_size=32
    server=Server(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    dest=tmp_path/'part01'; expected=hashlib.sha256(full).hexdigest()
    urls=[(f'http://127.0.0.1:{server.server_port}/{i}?private-signature=do-not-persist',1024) for i in range(12)]
    try:
        result=d.download_blocks(urls,len(full),expected,dest,12)
        assert result['workers']==12 and result['blocks']==12 and dest.read_bytes()==full
        assert d.download_blocks(urls,len(full),expected,dest,12)['status']=='already_verified'
        assert all('private-signature' not in p.read_text(errors='ignore') for p in tmp_path.iterdir())
    finally: server.shutdown(); server.server_close()
    # Resumed bytes are rechecked, and only the missing block is fetched.
    dest.unlink(); partial=dest.with_name(dest.name+'.partial'); partial.write_bytes(full)
    completed={str(i):hashlib.sha256(data[i]).hexdigest() for i in range(11)}
    dest.with_name(dest.name+'.resume.json').write_text(json.dumps({'sha256':expected,'bytes':len(full),'block_size':1024,'completed':completed}))
    class Response:
        status=200
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,n): return data[11]
    calls=[]
    monkeypatch.setattr(d.urllib.request,'urlopen',lambda url,timeout:(calls.append(url) or Response()))
    result=d.download_blocks(urls,len(full),expected,dest,12)
    assert result['resumed_blocks']==11 and len(calls)==1 and dest.read_bytes()==full


def test_wrong_final_hash_preserves_partial_and_resets_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'BLOCK',4)
    class Response:
        status=200
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,n): return b'bad'
    monkeypatch.setattr(d.urllib.request,'urlopen',lambda *args,**kwargs:Response())
    with pytest.raises(ValueError,match='whole part SHA256'):
        d.download_blocks([('https://example.invalid/block?signature=private-url',4)],4,hashlib.sha256(b'good').hexdigest(),tmp_path/'part',12)
    state=json.loads((tmp_path/'part.resume.json').read_text())
    assert state['completed']=={} and 'private-url' not in json.dumps(state)
