"""Own one Linux native process group; cancel only its verified PID identity."""
import json
import os
from pathlib import Path
import signal
import sys


def start_token(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def main():
    request=json.loads(sys.stdin.read())
    pid_file=Path(request['pid_path'])
    cancel=Path(request['cancel_path'])
    if request.get('action')=='cancel':
        cancel.touch()
        if not pid_file.exists():return
        state=json.loads(pid_file.read_text())
        try:
            pid=state['pid']
            if start_token(pid)!=state['start']:return
            args=Path(f'/proc/{pid}/cmdline').read_bytes().split(bytes([0]))
            expected={os.fsencode(request['binary']),os.fsencode(Path(__file__).resolve())}
            if not expected.intersection(args):raise RuntimeError('native worker identity differs')
            if os.getpgid(pid)!=pid:raise RuntimeError('native worker does not own its process group')
            os.killpg(pid,signal.SIGKILL)
        except (FileNotFoundError,ProcessLookupError):pass
        return
    if os.getpgrp()!=os.getpid():os.setsid()
    temporary=pid_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(pid=os.getpid(),start=start_token(os.getpid()))))
    os.replace(temporary,pid_file)
    if cancel.exists():return
    args=[request['binary'],'-s','2025','-o','tour.sol']
    if Path('case.res').exists():args.extend(['-R','case.res'])
    args.append('case.tsp')
    os.execv(request['binary'],args)


if __name__=='__main__':main()
