"""Verify and unpack the full two-round evidence snapshot (Python standard library)."""
import gzip,hashlib,json,os,sys,tempfile,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads(gzip.decompress((root/'manifest.json.gz').read_bytes()).decode('utf-8'))
dest=Path(sys.argv[1] if len(sys.argv)>1 else 'two_rounds_20261112').resolve()
if dest.exists() and any(dest.iterdir()):raise SystemExit('Destination must be empty')
whole=hashlib.sha256()
with tempfile.TemporaryFile() as stream:
    for part in manifest['parts']:
        path=root/part['path'];digest=hashlib.sha256();size=0
        with path.open('rb') as f:
            while data:=f.read(1024*1024):digest.update(data);whole.update(data);stream.write(data);size+=len(data)
        if digest.hexdigest()!=part['sha256'] or size!=part['bytes']:raise SystemExit('Invalid archive part: '+str(path))
    if whole.hexdigest()!=manifest['archive_sha256']:raise SystemExit('Archive checksum mismatch')
    stream.seek(0)
    with zipfile.ZipFile(stream) as z:
        for name in z.namelist():
            if not (dest/name).resolve().is_relative_to(dest):raise SystemExit('Unsafe member: '+name)
        for entry in manifest['entries']:
            if entry['path'].startswith('project/') and hashlib.sha256(z.read(entry['path'])).hexdigest()!=entry['sha256']:
                raise SystemExit('Member checksum mismatch: '+entry['path'])
        extract_dest=str(dest)
        if os.name=='nt':
            extract_dest=('\\\\?\\UNC\\'+extract_dest[2:] if extract_dest.startswith('\\\\') else '\\\\?\\'+extract_dest)
        z.extractall(extract_dest)
print('Verified evidence extracted to:',dest)
