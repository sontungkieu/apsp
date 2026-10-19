"""Real curl/proxy/redirect regression, bounded fixtures, no remote model downloads.

Run with existing Linux bash/curl: python3 tests/lite/test_hf_curl_template.py.
Windows pytest skips this case; the WSL execution is recorded separately.
"""
import hashlib,json,os,shutil,subprocess,tempfile,threading,time,unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path


@unittest.skipIf(os.name=='nt','Linux bash/curl protocol check; run separately in WSL')
class CurlTemplateTest(unittest.TestCase):
    def test_redirect_resume_twelve_ranges_and_refuse_ignored_range(self):
        root=Path(__file__).resolve().parents[2]
        body=bytes(range(251))*41
        status={'active':0,'peak':0,'ignore':False};guard=threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if 'redirect-final' not in self.path:
                    self.send_response(302);self.send_header('Location','http://fixture.invalid/redirect-final')
                    self.send_header('Content-Length','2048');self.end_headers()
                    try:self.wfile.write(b'x'*2048)
                    except (BrokenPipeError,ConnectionResetError):pass
                    return
                start,end=map(int,self.headers.get('Range','bytes=0-0').split('=')[1].split('-'))
                with guard:status['active']+=1;status['peak']=max(status['peak'],status['active'])
                try:
                    if end>start:time.sleep(.15)
                    payload=body if status['ignore'] else body[start:end+1]
                    self.send_response(200 if status['ignore'] else 206)
                    self.send_header('Content-Length',str(len(payload)));self.end_headers()
                    try:self.wfile.write(payload)
                    except (BrokenPipeError,ConnectionResetError):pass
                finally:
                    with guard:status['active']-=1
            def log_message(self,*args):pass
        class Server(ThreadingHTTPServer):request_queue_size=32
        server=Server(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            with tempfile.TemporaryDirectory(prefix='apsp-curl-fixture-') as temp:
                work=Path(temp);dest=work/'download';dest.mkdir()
                manifest=work/'parts.json';digest=hashlib.sha256(body).hexdigest()
                manifest.write_text(json.dumps({'parts':[{'file':'fixture.part01','bytes':len(body),'sha256':digest}]}))
                ranges=dest/'fixture.part01.ranges';ranges.mkdir()
                (ranges/'contract').write_text(f'fixture.part01 {len(body)} {digest} 12\n')
                (ranges/'range.00').write_bytes(body[:13])
                shim=work/'bin';shim.mkdir();tool=shim/'curl'
                tool.write_text("#!/usr/bin/env python3\nimport os,sys\na=[x.replace('https://huggingface.co/','http://fixture.invalid/') if x.startswith('https://huggingface.co/') else x for x in sys.argv[1:]]\nos.execv('/usr/bin/curl',['curl',*a])\n")
                tool.chmod(0o755)
                env={**os.environ,'PROXY':f'http://127.0.0.1:{server.server_port}','PATH':str(shim)+os.pathsep+os.environ['PATH'],'APSP_DOWNLOAD_THREADS':'12'}
                def run(target):
                    return subprocess.run(['bash',str(root/'env/lite/download_part_hf.sh'),str(manifest),'1',str(target),'fixture/repo','a'*40],env=env,capture_output=True,text=True,timeout=60)
                result=run(dest)
                self.assertEqual(result.returncode,0,result.stdout)
                self.assertEqual((dest/'fixture.part01').read_bytes(),body)
                self.assertEqual(status['peak'],12)
                self.assertEqual(run(dest).returncode,0)
                status['ignore']=True
                result=run(work/'bad-server')
                self.assertNotEqual(result.returncode,0)
                self.assertFalse((work/'bad-server/fixture.part01').exists())
                print('HF_CURL_REGRESSION_JSON='+json.dumps({'pass':True,'bytes':len(body),'concurrent_requests':status['peak'],'resume_from_bytes':13,'uneven_final_range':True,'redirect_body_larger_than_range':True,'ignored_range_refused':True,'scope':'Bounded Linux curl fixture, not company proxy/node or throughput benchmark.'}))
        finally:server.shutdown();server.server_close()


if __name__=='__main__':unittest.main()
