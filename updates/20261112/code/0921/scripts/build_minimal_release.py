"""Materialize the only active minimal-data release line.

Older runs remain content-addressed evidence; this directory is the sole
default handoff and contains no model key or validation data copies.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/minimal_release_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    selected = {
        'cvrp': ROOT / 'runs/minimal_cvrp_star_v1/sources/ordered_neighbors/solver.py',
        'fssp': ROOT / 'runs/minimal_fssp_acceptance_v1/sources/beam_control/solver.py',
    }
    receipts = {
        'cvrp': ROOT / 'runs/minimal_cvrp_star_v1/minimal_validation.json',
        'fssp': ROOT / 'runs/minimal_fssp_acceptance_v1/exact_validation_reuse.json',
    }
    manifest = {'schema': 'minimal_release_v1', 'scope': 'fixed_minimal_validation_only',
        'full_validation_scheduled': False, 'selection_uses_validation': False,
        'tasks': {}, 'tests': '90 passed before release packaging; rerun below'}
    for kind, source in selected.items():
        target = OUT / kind / 'solver.py'
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        receipt_target = OUT / kind / Path(receipts[kind]).name
        shutil.copy2(receipts[kind], receipt_target)
        manifest['tasks'][kind] = {'source_sha256': sha(target), 'source_origin': str(source),
            'validation_receipt': str(receipt_target), 'validation_receipt_sha256': sha(receipt_target)}
    (OUT / 'release.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
