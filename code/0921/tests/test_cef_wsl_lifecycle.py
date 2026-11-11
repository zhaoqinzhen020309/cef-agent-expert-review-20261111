"""Cancel the real native Linux process, not just its Windows launcher."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import time
import unittest
import numpy as np


class NativeLifecycleTests(unittest.TestCase):
    def test_real_concorde_process_is_gone_after_cancellation(self):
        from cef.reference_native import ROOT,VENDOR,WSL_DISTRIBUTION,linux_path,cancel_native,prepare_problem
        from cef.store import canonical
        fixture=ROOT/'acceptance/scip_replay_20261103/supplement_0040.npz'
        d=np.load(fixture)['distance_matrix'];p=prepare_problem(d)
        with tempfile.TemporaryDirectory(prefix='cef_cancel_') as td:
            work=Path(td)
            (work/'case.tsp').write_text('NAME: case\nTYPE: TSP\nDIMENSION: 100\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n'+'\n'.join(' '.join(map(str,r)) for r in p['integer_matrix'])+'\nEOF\n',encoding='ascii')
            request=dict(binary=linux_path(VENDOR/'concorde'),pid_path=linux_path(work/'pid.json'),cancel_path=linux_path(work/'cancel.flag'),cancel_name='cancel.flag')
            with (work/'out.log').open('w') as out,(work/'err.log').open('w') as err:
                process=subprocess.Popen(['wsl.exe','-d',WSL_DISTRIBUTION,'--cd',linux_path(work),'--exec','python3',linux_path(ROOT/'cef/reference_native_worker.py')],stdin=subprocess.PIPE,stdout=out,stderr=err,text=True)
                try:
                    process.stdin.write(canonical(request));process.stdin.close()
                    deadline=time.monotonic()+10
                    while not (work/'pid.json').exists() and time.monotonic()<deadline:time.sleep(.05)
                    pid=json.loads((work/'pid.json').read_text())['pid']
                    time.sleep(.3)
                    self.assertIsNone(process.poll(),'test must cancel an active native search')
                finally:cancel_native(process,request,work)
            check=subprocess.run(['wsl.exe','-d',WSL_DISTRIBUTION,'--exec','python3','-c',
                "from pathlib import Path;p=Path('/proc/"+str(pid)+"/stat');assert not p.exists() or p.read_text().rsplit(')',1)[1].split()[0]=='Z'"],capture_output=True,timeout=10)
            self.assertEqual(check.returncode,0,check.stderr)
            self.assertIsNotNone(process.poll())
