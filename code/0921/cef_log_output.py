"""Console presentation and append-only logs; never changes experiment decisions."""
import ast
import hashlib
import json
import re
import sys
import threading
import time
import traceback
from pathlib import Path

from watchCEF import JournalTail, LogView, journals

OLD = "if __name__=='__main__':\n from scripts.cef_alternating_launcher import main\n raise SystemExit(main(CONFIG))\n"
NEW = "if __name__=='__main__':\n from cef_log_output import launch\n raise SystemExit(launch(CONFIG))\n"


def equivalent_entrypoints(old, new):
    """Allow only the exact presentation bootstrap replacement, not CONFIG edits."""
    try:
        a, b = ast.parse(old), ast.parse(new)
        if ast.dump(a.body[-1]) != ast.dump(ast.parse(OLD).body[0]): return False
        if ast.dump(b.body[-1]) != ast.dump(ast.parse(NEW).body[0]): return False
        return [ast.dump(n) for n in a.body[:-1]] == [ast.dump(n) for n in b.body[:-1]]
    except (SyntaxError, IndexError):
        return False


class _Stream:
    def __init__(self, owner, original):
        self.owner, self.original, self.pending = owner, original, ''
    def write(self, text):
        with self.owner.lock:
            self.owner.raw(text)
            self.pending += text
            while '\n' in self.pending:
                line, self.pending = self.pending.split('\n', 1)
                if not re.match(r'^(data_designer|planner|editor) \| response=\d+|^Experiment \| |^(Data_Designer|Planner|Editor) task \|', line):
                    self.owner.emit(line+'\n', self.original)
        return len(text)
    def flush(self):
        self.original.flush()
    def finish(self):
        if self.pending:
            self.owner.emit(self.pending, self.original); self.pending = ''
    def __getattr__(self, name): return getattr(self.original, name)


class RunLog:
    """Single-console tee plus read-only semantic journal rendering."""
    def __init__(self, folder, visible=True):
        self.folder = Path(folder); self.visible = visible
        self.lock = threading.RLock(); self.stop = threading.Event()
        self.tails = {}; self.view = LogView(folder); self.cursor = {}; self.files = None
        self.buffer = []; self.error = None
    def open_files(self):
        # The launcher must create its own new folder with exist_ok=False.
        if self.files is not None or not self.folder.is_dir(): return
        from cef.cli import claim_owner
        control = self.folder/'logging_control'; control.mkdir(exist_ok=True)
        self.lease = claim_owner(control)
        try:
            path = self.folder/'progress.cursor.json'
            if path.exists(): self.cursor = json.loads(path.read_text(encoding='utf-8'))
            self.files = {n:(self.folder/n).open('a', encoding='utf-8', newline='\n', buffering=1)
                          for n in ('progress.log','console.raw.log')}
            sources=[Path(__file__),Path(__file__).with_name('watchCEF.py'),Path(__file__).with_name('runCEFAuto.py')]
            manifest=dict(scope='presentation only; no experiment configuration changes',
                          sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])
            (self.folder/'logging_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8',newline='\n')
            for name,text in self.buffer: self.files[name].write(text)
            self.buffer.clear()
        except BaseException:
            self.lease.unlink(missing_ok=True); raise
    def raw(self, text):
        self.open_files()
        if self.files: self.files['console.raw.log'].write(text)
        else: self.buffer.append(('console.raw.log',text))
    def emit(self, text, terminal=None):
        self.open_files()
        if self.visible: (terminal or self.stdout).write(text); (terminal or self.stdout).flush()
        if self.files: self.files['progress.log'].write(text)
        else: self.buffer.append(('progress.log',text))
    def poll(self):
        with self.lock:
            self.open_files()
            if not self.files: return
            rows=[]
            for path in journals(self.folder):
                tail=self.tails.setdefault(path,JournalTail(path))
                rows.extend((r,path) for r in tail.poll())
            rows.sort(key=lambda x:(x[0].get('time',0),str(x[1]),x[0].get('seq',0)))
            for record,path in rows:
                key=path.relative_to(self.folder).as_posix(); seq=record['seq']
                lines=self.view.consume(path,record)
                if seq>self.cursor.get(key,-1):
                    for line in lines: self.emit(line+'\n')
                    self.cursor[key]=seq
            if rows:
                self.files['progress.log'].flush()
                temporary=self.folder/'progress.cursor.tmp'
                temporary.write_text(json.dumps(self.cursor),encoding='utf-8',newline='\n')
                temporary.replace(self.folder/'progress.cursor.json')
    def loop(self):
        while not self.stop.wait(1):
            try: self.poll()
            except Exception as exc:
                self.error=exc
                self.stderr.write('CEF logging error: '+repr(exc)+'\n'); self.stderr.flush()
                return
    def __enter__(self):
        self.stdout,self.stderr=sys.stdout,sys.stderr
        self.poll()
        if self.visible:
            self.out,self.err=_Stream(self,self.stdout),_Stream(self,self.stderr)
            sys.stdout,sys.stderr=self.out,self.err
        self.thread=threading.Thread(target=self.loop,name='cef-progress-log',daemon=True)
        self.thread.start(); return self
    def __exit__(self, kind, value, tb):
        self.stop.set(); self.thread.join()
        try:
            self.poll()
            if self.visible: self.out.finish(); self.err.finish()
            if kind:
                text=''.join(traceback.format_exception(kind,value,tb))
                self.raw(text)
                if self.files: self.files['progress.log'].write(text)
        finally:
            if self.visible: sys.stdout,sys.stderr=self.stdout,self.stderr
            if self.files:
                for stream in self.files.values(): stream.close()
                self.lease.unlink(missing_ok=True)
        if self.error is not None and kind is None:
            raise RuntimeError('CEF progress logging failed; experiment records retained') from self.error
        return False


def launch(settings, argv=None):
    import scripts.cef_alternating_launcher as launcher
    original_run, original_validate = launcher.run_folder, launcher.validate
    run_entered=False
    def validate(config, *args, **kwargs):
        manifest=config.get('implementation_source_manifest',[])
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        changed=[e for e in manifest if digest(e['path'])!=e['sha256']]
        if changed:
            entry=Path(__file__).with_name('runCEFAuto.py')
            saved=Path(config['run_dir'])/'runtime_snapshot/runCEFAuto.py'
            if (len(changed)!=1 or Path(changed[0]['path']).resolve()!=entry.resolve()
                or not saved.is_file() or digest(saved)!=changed[0]['sha256']
                or not equivalent_entrypoints(saved.read_text(encoding='utf-8-sig'),entry.read_text(encoding='utf-8-sig'))):
                return original_validate(config,*args,**kwargs)
            result=original_validate(dict(config,implementation_source_manifest=[]),*args,**kwargs)
            result['implementation_source_manifest']=manifest
            receipt=dict(scope='exact logging bootstrap replacement; all other source hashes checked',
                         frozen_sha256=changed[0]['sha256'],current_sha256=digest(entry))
            (Path(config['run_dir']).parent/'logging_compatibility.json').write_text(
                json.dumps(receipt,indent=2),encoding='utf-8',newline='\n')
            return result
        return original_validate(config,*args,**kwargs)
    def run(config,test,folder,resume=False,**kwargs):
        nonlocal run_entered
        with RunLog(folder):
            run_entered=True
            return original_run(config,test,folder,resume,**kwargs)
    launcher.run_folder,launcher.validate=run,validate
    try:
        return launcher.main(settings,argv)
    except Exception:
        if not run_entered:
            try:
                folder=Path(settings['output_dir'])/'startup_errors'
                folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'startup_{time.time_ns()}.log'
                path.write_text(traceback.format_exc(),encoding='utf-8',newline='\n')
                print('Startup error log: '+str(path),file=sys.stderr,flush=True)
            except Exception as logging_error:
                print('Could not save startup error: '+str(logging_error),file=sys.stderr,flush=True)
        raise
    finally: launcher.run_folder,launcher.validate=original_run,original_validate


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Backfill readable CEF journal logs without model calls.')
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    if not (args.run/'search_config.json').is_file(): parser.error('run configuration not found')
    with RunLog(args.run,visible=False): pass
    print(args.run/'progress.log')


if __name__=='__main__': main()
