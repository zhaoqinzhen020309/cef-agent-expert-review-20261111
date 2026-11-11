"""Verify and unpack the complete round-1 trace using only Python's standard library."""
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'round1_trace')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'manifests/EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
    output=args.output.resolve()
    if not args.verify_only and output.exists() and any(output.iterdir()):
        raise SystemExit('Output directory is not empty. Choose a new --output directory.')
    full=hashlib.sha256()
    with tempfile.TemporaryFile() as joined:
        for part in manifest['archive_parts']:
            path=ROOT/part['path'];h=hashlib.sha256();count=0
            with path.open('rb') as f:
                while data:=f.read(1024*1024):
                    h.update(data);full.update(data);joined.write(data);count+=len(data)
            if count!=part['bytes'] or h.hexdigest()!=part['sha256']:
                raise SystemExit('Part checksum mismatch: '+part['path'])
            print('Verified '+part['path'],flush=True)
        if full.hexdigest()!=manifest['zip_sha256']:raise SystemExit('Full archive checksum mismatch')
        joined.seek(0)
        with zipfile.ZipFile(joined) as archive:
            expected={r['path']:r for r in manifest['entries'] if r['container']=='artifacts/round_001_full_trace.zip'}
            if set(archive.namelist())!=set(expected):raise SystemExit('Archive file list differs from manifest')
            for index,info in enumerate(archive.infolist(),1):
                path=(output/info.filename).resolve()
                if not path.is_relative_to(output):raise SystemExit('Unsafe archive path')
                h=hashlib.sha256()
                with archive.open(info) as src:
                    if args.verify_only:
                        while data:=src.read(1024*1024):h.update(data)
                    else:
                        path.parent.mkdir(parents=True,exist_ok=True)
                        with path.open('xb') as dst:
                            while data:=src.read(1024*1024):h.update(data);dst.write(data)
                if h.hexdigest()!=expected[info.filename]['published_sha256']:
                    raise SystemExit('Extracted file checksum mismatch: '+info.filename)
                if index%200==0:print(f'Checked {index}/{len(expected)} files',flush=True)
    print('All round-1 files verified.' if args.verify_only else 'Extracted to '+str(output))

if __name__=='__main__':main()
