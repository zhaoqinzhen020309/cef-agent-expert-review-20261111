"""Windows process pause at an idle external-work boundary; no frozen code edits."""
import argparse
import ctypes
from ctypes import wintypes as w
from datetime import datetime
import json
from pathlib import Path
import time

k=ctypes.WinDLL('kernel32',use_last_error=True)
n=ctypes.WinDLL('ntdll')
k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];k.OpenProcess.restype=w.HANDLE
k.CloseHandle.argtypes=[w.HANDLE];k.CloseHandle.restype=w.BOOL
k.GetProcessTimes.argtypes=[w.HANDLE,ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME)]
k.GetExitCodeProcess.argtypes=[w.HANDLE,ctypes.POINTER(w.DWORD)]
k.QueryFullProcessImageNameW.argtypes=[w.HANDLE,w.DWORD,w.LPWSTR,ctypes.POINTER(w.DWORD)]
n.NtSuspendProcess.argtypes=[w.HANDLE];n.NtSuspendProcess.restype=ctypes.c_long
n.NtResumeProcess.argtypes=[w.HANDLE];n.NtResumeProcess.restype=ctypes.c_long


class Entry(ctypes.Structure):
    _fields_=[('dwSize',w.DWORD),('cntUsage',w.DWORD),('th32ProcessID',w.DWORD),
        ('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',w.DWORD),('cntThreads',w.DWORD),
        ('th32ParentProcessID',w.DWORD),('pcPriClassBase',w.LONG),('dwFlags',w.DWORD),('szExeFile',w.WCHAR*260)]
k.CreateToolhelp32Snapshot.argtypes=[w.DWORD,w.DWORD];k.CreateToolhelp32Snapshot.restype=w.HANDLE
k.Process32FirstW.argtypes=[w.HANDLE,ctypes.POINTER(Entry)];k.Process32NextW.argtypes=[w.HANDLE,ctypes.POINTER(Entry)]


def opened(pid):
    h=k.OpenProcess(0x1000|0x800|0x100000,False,pid)
    if not h:raise ctypes.WinError(ctypes.get_last_error())
    return h


def identity(handle):
    creation,exit_time,kernel,user=w.FILETIME(),w.FILETIME(),w.FILETIME(),w.FILETIME()
    if not k.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(kernel),ctypes.byref(user)):raise ctypes.WinError(ctypes.get_last_error())
    name=ctypes.create_unicode_buffer(32768);size=w.DWORD(len(name))
    if not k.QueryFullProcessImageNameW(handle,0,name,ctypes.byref(size)):raise ctypes.WinError(ctypes.get_last_error())
    return dict(creation=creation.dwHighDateTime*2**32+creation.dwLowDateTime,executable=name.value)


def descendants(pid):
    snapshot=k.CreateToolhelp32Snapshot(2,0)
    if snapshot==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    entries=[];entry=Entry();entry.dwSize=ctypes.sizeof(entry)
    try:
        ok=k.Process32FirstW(snapshot,ctypes.byref(entry))
        while ok:
            entries.append((entry.th32ProcessID,entry.th32ParentProcessID,entry.szExeFile))
            ok=k.Process32NextW(snapshot,ctypes.byref(entry))
    finally:k.CloseHandle(snapshot)
    found={pid};result=[]
    while True:
        children=[x for x in entries if x[1] in found and x[0] not in found]
        if not children:break
        for child,parent,name in children:
            found.add(child)
            # A console host is not a solver. Verify the system executable path.
            if name.lower()=='conhost.exe':
                h=k.OpenProcess(0x1000,False,child)
                try:
                    if h and Path(identity(h)['executable']).resolve()==Path('C:/Windows/System32/conhost.exe').resolve():continue
                finally:
                    if h:k.CloseHandle(h)
            result.append(dict(pid=child,parent=parent,name=name))
    return result


class Tail:
    def __init__(self,path):
        self.path=path;self.offset=0;self.buffer=b'';self.requests=0;self.responses=0
        self.started=set();self.done=set();self.rounds=[]
    def update(self):
        with self.path.open('rb') as stream:
            stream.seek(self.offset);chunk=stream.read();self.offset=stream.tell()
        self.buffer+=chunk
        lines=self.buffer.split(b'\n');self.buffer=lines.pop()
        for line in lines:
            if not line:continue
            x=json.loads(line);kind=x['kind'];value=x['data']
            if kind=='provider_request':self.requests+=1
            if kind=='provider_response':self.responses+=1
            if kind in ('execution_start','observation_start'):self.started.add(value.get('execution_id',value.get('key')))
            if kind in ('execution','probe_execution','observation'):self.done.add(value.get('execution_id',value.get('key')))
            if kind=='round_complete':self.rounds.append(value['round'])
    def quiet(self):return not self.buffer and self.requests==self.responses and not self.started-self.done


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path);p.add_argument('--pid',type=int)
    p.add_argument('--after-round',type=int,default=0)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--resume',type=Path)
    args=p.parse_args(argv)
    if args.resume:
        saved=json.loads(args.resume.read_text(encoding='utf-8'));h=opened(saved['pid'])
        try:
            if identity(h)!=saved['process_identity']:raise RuntimeError('process identity differs; not resuming')
            if n.NtResumeProcess(h)!=0:raise RuntimeError('NtResumeProcess failed')
            args.resume.with_suffix('.resumed.json').write_text(json.dumps(dict(resumed_at=datetime.now().isoformat(),pid=saved['pid'])),encoding='utf-8')
            print('Resumed original process | '+str(saved['pid']),flush=True)
        finally:k.CloseHandle(h)
        return
    if not args.run or not args.pid or not args.receipt:p.error('--run --pid --receipt required')
    run=args.run.resolve()
    if int((run/'search/owner.lock').read_text())!=args.pid:raise RuntimeError('run owner differs')
    h=opened(args.pid);birth=identity(h)
    tails=[Tail(run/f'{stage}/facts.jsonl') for stage in ('search','data_agent','measurements')]
    deadline=time.monotonic()+3600;last=0
    try:
        while time.monotonic()<deadline:
            code=w.DWORD()
            if not k.GetExitCodeProcess(h,ctypes.byref(code)) or code.value!=259:
                print('Experiment already exited; no process paused',flush=True);return
            for tail in tails:tail.update()
            if all(t.quiet() for t in tails) and len(tails[0].rounds)>=args.after_round and not descendants(args.pid):
                if n.NtSuspendProcess(h)!=0:raise RuntimeError('NtSuspendProcess failed')
                accepted=False
                try:
                    for tail in tails:tail.update()
                    accepted=all(t.quiet() for t in tails) and not descendants(args.pid)
                    if accepted:
                        saved=dict(pid=args.pid,process_identity=birth,run=str(run),paused_at=datetime.now().isoformat(),
                            completed_rounds=tails[0].rounds,
                            completed_observations=len(tails[2].done),pending_api_requests=0,pending_solver_executions=0,
                            method='NtSuspendProcess; original memory, configuration and owner lock retained')
                        args.receipt.parent.mkdir(parents=True,exist_ok=True)
                        args.receipt.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
                        print('PAUSED | '+json.dumps(saved,ensure_ascii=False),flush=True);return
                finally:
                    if not accepted:n.NtResumeProcess(h)
            if time.monotonic()-last>30:
                print(f'Waiting safe boundary | completed_rounds={len(tails[0].rounds)} | pending_api={sum(t.requests-t.responses for t in tails)} | pending_solver={len(tails[2].started-tails[2].done)}',flush=True)
                last=time.monotonic()
            time.sleep(.02)
        raise RuntimeError('Safe-boundary monitor timed out; experiment left running')
    finally:k.CloseHandle(h)


if __name__=='__main__':main()
