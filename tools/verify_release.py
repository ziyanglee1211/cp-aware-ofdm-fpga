"""Check a release file manifest using only Python's standard library."""
from pathlib import Path
import hashlib,json,sys

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]
    manifest=root/'provenance/release_manifest.json'
    if not manifest.exists():raise SystemExit('Missing release_manifest.json; finalize or extract the complete distribution.')
    spec=json.loads(manifest.read_text(encoding='utf-8'))
    failures=[]
    for row in spec['files']:
        p=(root/row['path']).resolve()
        p.relative_to(root)
        if not p.is_file() or sha(p)!=row['sha256']:failures.append(row['path'])
    known={r['path'] for r in spec['files']}|{'provenance/release_manifest.json'}
    extras=[p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in known and not any(x in p.parts for x in ['.git','__pycache__','generated_figures','generated_runs'])]
    print(json.dumps({'status':'fail' if failures else 'pass','manifest_kind':spec['distribution'],'checked_files':len(spec['files']),'failed_files':failures,'unmanifested_files':extras,'scope':'Integrity only, not numerical or scientific verification.'}))
    if failures:sys.exit(1)

if __name__=='__main__':main()
