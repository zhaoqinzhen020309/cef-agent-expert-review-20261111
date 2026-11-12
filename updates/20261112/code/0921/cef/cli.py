"""Executable CEF mode; validate before creating a run or making model calls."""
import argparse,json,os,sys
import shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from .config import load_json,validate
from .store import Archive,canonical
from .engine import Engine
from .model import Model
from .data import prepare,audit_training,audit_test
from .reporting import verify_export,final_test,import_external

def load_env(path):
    if path is None: return
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        k,sep,v=line.partition('=')
        if not sep: raise ValueError('environment requires KEY=value')
        os.environ.setdefault(k.strip(),v.strip().strip('\"\''))

def claim_owner(root):
    lock=Path(root)/'owner.lock'
    if lock.exists():
        pid=int(lock.read_text(encoding='ascii').strip())
        if pid<=0: raise ValueError('invalid ownership receipt')
        if sys.platform=='win32':
            import ctypes
            k=ctypes.WinDLL('kernel32',use_last_error=True)
            k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]; k.OpenProcess.restype=ctypes.c_void_p
            k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
            k.CloseHandle.argtypes=[ctypes.c_void_p]
            h=k.OpenProcess(0x1000,False,pid)
            if not h:
                if ctypes.get_last_error()!=87: raise FileExistsError('cannot prove previous owner is dead')
                alive=False
            else:
                code=ctypes.c_ulong()
                try:
                    if not k.GetExitCodeProcess(h,ctypes.byref(code)): raise FileExistsError('owner status unavailable')
                    alive=code.value==259
                finally: k.CloseHandle(h)
        else:
            try: os.kill(pid,0); alive=True
            except ProcessLookupError: alive=False
        if alive: raise FileExistsError('CEF run already owned by a live process')
        lock.rename(Path(root)/f'owner.abandoned.{pid}.{time.time_ns()}.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w',encoding='ascii') as f:
        f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno())
    return lock

def freeze_runtime(root,config):
    folder=Path(root)/'runtime_snapshot'
    for entry in config.get('implementation_source_manifest',[]):
        src=Path(entry['path'])
        relative=src.relative_to(Path(__file__).resolve().parents[1])
        dst=folder/relative; dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            from .config import file_hash
            if file_hash(dst)!=entry['sha256']: raise ValueError('frozen runtime source corrupted')
        else: shutil.copyfile(src,dst)

def main(argv=None):
    p=argparse.ArgumentParser(prog='0921 CEF-Agent')
    p.add_argument('--quiet',action='store_true',help='suppress final JSON on stdout; evidence remains in files')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('validate','run','resume','preflight'):
        s=sub.add_parser(name); s.add_argument('--config',required=True); s.add_argument('--env-file')
    s=sub.add_parser('prepare-data'); s.add_argument('--output',required=True); s.add_argument('--data-seed',type=int,required=True)
    s=sub.add_parser('status'); s.add_argument('--run-dir',required=True)
    s=sub.add_parser('test'); s.add_argument('--run-dir',required=True); s.add_argument('--test-config',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('import-external'); s.add_argument('--manifest',required=True); s.add_argument('--results',required=True); s.add_argument('--output',required=True)
    args=p.parse_args(argv)
    try:
        if args.command=='prepare-data': result=prepare(Path(__file__).resolve().parents[1],args.output,args.data_seed)
        elif args.command=='status':
            a=Archive(args.run_dir); result=dict(candidates=len(a.of('candidate')),executions=len(a.of('execution')),events=len(a.of('contrast')),last_termination=a.of('termination')[-1:] )
        elif args.command=='test':
            t=load_json(args.test_config); a,result,_=verify_export(args.run_dir)
            if t['test_repeats']!=a.of('manifest')[0]['test_repeats']: raise ValueError('test repeats differ from frozen run')
            result=final_test(args.run_dir,args.output,t['instances'],t['test_repeats'],t['evaluation_seeds'])
        elif args.command=='import-external': result=import_external(args.manifest,args.results,args.output)
        else:
            raw=load_json(args.config)
            c=validate(raw,formal=not raw.get('diagnostic_run',False)); audit_training(c['task'],c['data_manifest']['train'],c['data_manifest'].get('training_protocol')); audit_test(c['task'],c['data_manifest']['test'])
            if args.command=='validate': result=dict(status='valid',model=c['model_snapshot'],mode=c['mode'],task=c['task'])
            else:
                if c.get('alternating_agents') and args.command in ('run','resume'):
                    raise ValueError('Use runCEFAuto.py for alternating search; fixed-data CLI cannot execute this protocol')
                load_env(args.env_file)
                if not os.environ.get(c['api_key_env']): raise ValueError('authentication not configured in '+c['api_key_env'])
                root=Path(c['run_dir'])
                if args.command=='run' and root.exists(): raise FileExistsError('new run directory must not exist')
                if args.command=='resume' and not (root/'facts.jsonl').exists(): raise FileNotFoundError('resume requires actual journal')
                if args.command=='preflight':
                    from .preflight import preflight
                    result=preflight(c)
                else:
                    gate=load_json(c['connectivity_receipt'])
                    from .preflight import receipt_matches
                    if not receipt_matches(c,gate): raise ValueError('matching complete-code preflight required')
                    # One active owner, including resume. A stale lease is diagnosed,
                    # never silently stolen from a potentially active process.
                    root.mkdir(parents=True,exist_ok=True)
                    lock=claim_owner(root)
                    try:
                        freeze_runtime(root,c)
                        e=Engine(c,None); e.model=Model(c,e.archive,e.ledger)
                        result=e.run()
                    finally: lock.unlink()
        if not args.quiet: print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except Exception as e:
        print(json.dumps(dict(status='error',error=type(e).__name__+': '+str(e)),ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__': raise SystemExit(main())
