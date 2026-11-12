"""Keep the native VM alive across long model/tool phases, without polling the API."""
import queue
import subprocess
import threading
import time
import json
from pathlib import Path


class NativeBackendSession:
    def __init__(self,command=None,event_path=None):
        self.command=command;self.event_path=Path(event_path) if event_path else None
    def record(self,kind,attempt,error=None):
        if self.event_path:
            self.event_path.parent.mkdir(parents=True,exist_ok=True)
            with self.event_path.open('a',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps(dict(time=time.time(),kind=kind,attempt=attempt,error=error),ensure_ascii=False)+'\n')
    def __enter__(self):
        attempt=0
        while True:
            attempt+=1
            try:
                result=self.start_once()
                self.record('ready',attempt)
                return result
            except RuntimeError as error:
                if 'Wsl/Service/CreateInstance/CreateVm/HCS/ERROR_NO_SYSTEM_RESOURCES' not in str(error):raise
                self.record('waiting_resources',attempt,str(error))
                print(f'WSL资源不足，等待5秒后重试启动 | attempt={attempt} | 尚未开始实验 | Ctrl+C可取消',flush=True)
                # This is environment startup only: no LLM request, search turn,
                # tool effect, alternate backend, or scientific retry occurs.
                time.sleep(5)
    def start_once(self):
        from cef.reference_native import WSL_DISTRIBUTION,decode_native_output
        command=self.command or ['wsl.exe','-d',WSL_DISTRIBUTION,'--exec','python3','-u','-c',
            "import sys;print('CEF_BACKEND_READY',flush=True);sys.stdin.buffer.read()"]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        ready=queue.Queue()
        def read():
            try:ready.put(self.process.stdout.readline())
            except Exception as error:ready.put(error)
        thread=threading.Thread(target=read,name='cef-native-startup',daemon=True);thread.start()
        try:
            line=ready.get(timeout=30)
            if isinstance(line,Exception):raise line
            if line.strip()!=b'CEF_BACKEND_READY':
                if self.process.poll() is None:self.process.kill()
                out,err=self.process.communicate(timeout=10)
                raise RuntimeError('WSL backend session could not start: '+decode_native_output(line+out+b'\n'+err))
        except BaseException:
            if self.process.poll() is None:self.process.kill()
            self.process.wait(timeout=10)
            for stream in (self.process.stdin,self.process.stdout,self.process.stderr):stream.close()
            raise
        return self
    def __exit__(self,kind,value,tb):
        try:
            was_alive=self.process.poll() is None
            try:self.process.stdin.close()
            except BrokenPipeError:pass
            self.process.stdin=None
            try:out,err=self.process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill();out,err=self.process.communicate(timeout=10)
                raise RuntimeError('WSL backend session did not release after run')
            if not was_alive or self.process.returncode:
                from cef.reference_native import decode_native_output
                raise RuntimeError('WSL backend session ended unexpectedly: '+decode_native_output(out+b'\n'+err))
        except Exception as cleanup:
            if value is not None:value.add_note(str(cleanup))
            else:raise
        return False
