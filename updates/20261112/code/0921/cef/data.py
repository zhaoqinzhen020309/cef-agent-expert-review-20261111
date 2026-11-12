"""Data import audits. Filename similarity is never a provenance check."""
import json,pickle,random
from pathlib import Path
import numpy as np
from .config import file_hash,load_json

TSPLIB29='eil51 berlin52 st70 eil76 pr76 rat99 kroA100 kroB100 kroC100 kroD100 kroE100 rd100 eil101 lin105 pr107 pr124 bier127 ch130 pr136 pr144 ch150 kroA150 kroB150 pr152 u159 rat195 d198 kroA200 kroB200'.split()
SCALES=[(20,5),(20,10),(20,20),(50,5),(50,10),(50,20),(100,5),(100,10),(100,20),(200,10),(200,20)]
TSPLIB_METRIC='tsplib_original_distance_v1'
TSPLIB_DISTANCE_RULE='original TSPLIB EUC_2D: floor(Euclidean edge length + 0.5); closed-tour sum'
TSPLIB_REFERENCE_RULE='frozen original published reference values under original TSPLIB distance rules'
UNROUNDED_TSPLIB_METRIC='tsplib_unrounded_original_reference_v1'
UNROUNDED_DISTANCE_RULE='unrounded Euclidean edge lengths; closed-tour sum'
UNROUNDED_REFERENCE_RULE='frozen original reference values; not re-certified as continuous-distance optima; negative gaps retained'
SUPPLEMENT_PROTOCOL='cef_tsp_original64_supplement64_v1'

def receipt(path,**extra): return dict(path=str(Path(path).resolve()),sha256=file_hash(path),**extra)

def prepare_tsplib_unrounded(root,rows,output):
    """Bind fresh unrounded matrices to original coordinates and frozen references."""
    root,output=Path(root),Path(output)
    converted=[]
    for row in rows:
        identifier=row['id']
        if Path(identifier).name!=identifier: raise ValueError('invalid TSPLIB identifier')
        original=Path(row['path'])
        if file_hash(original)!=row['sha256']: raise ValueError('original matrix hash mismatch: '+identifier)
        raw=root/'data/raw/tsplib'/f'{identifier}.tsp'
        parts=raw.read_text(encoding='utf-8').split('NODE_COORD_SECTION')
        if len(parts)!=2: raise ValueError('coordinate section required: '+identifier)
        header={k.strip():v.strip() for line in parts[0].splitlines() if ':' in line for k,v in [line.split(':',1)]}
        if header.get('EDGE_WEIGHT_TYPE')!='EUC_2D': raise ValueError('unrounded TSPLIB requires EUC_2D coordinates: '+identifier)
        n=int(header['DIMENSION']); coordinates={}
        for line in parts[1].splitlines():
            values=line.split()
            if not values or values[0]=='EOF': continue
            if len(values)!=3 or int(values[0]) in coordinates: raise ValueError('invalid coordinate row: '+identifier)
            coordinates[int(values[0])]=[float(values[1]),float(values[2])]
        if set(coordinates)!=set(range(1,n+1)): raise ValueError('incomplete coordinates: '+identifier)
        points=np.asarray([coordinates[i] for i in range(1,n+1)],dtype=float)
        if not np.isfinite(points).all(): raise ValueError('nonfinite coordinates: '+identifier)
        matrix=np.sqrt(((points[:,None,:]-points[None,:,:])**2).sum(axis=-1))
        with np.load(original,allow_pickle=False) as z:
            if not np.array_equal(z['distance_matrix'],np.floor(matrix+.5)):
                raise ValueError('coordinates do not match original matrix: '+identifier)
        output.mkdir(parents=True,exist_ok=True); target=output/f'{identifier}.npz'
        if target.exists():
            with np.load(target,allow_pickle=False) as z:
                if not np.array_equal(z['distance_matrix'],matrix): raise ValueError('unrounded matrix changed: '+identifier)
        else: np.savez(target,distance_matrix=matrix)
        converted.append(dict(row,**receipt(target),evaluation_metric=UNROUNDED_TSPLIB_METRIC,
                              reference_kind=UNROUNDED_REFERENCE_RULE,
                              source=dict(original=row.get('source'),original_matrix=receipt(original),coordinates=receipt(raw),distance_rule=UNROUNDED_DISTANCE_RULE)))
    return converted


def bind_original_tsplib(manifest):
    """New-run binding to original matrices; training and historical files stay intact."""
    import copy
    result=copy.deepcopy(manifest)
    rows=[];root=Path(__file__).resolve().parents[1]
    for row in result['test']:
        binding=row.get('source',{}).get('original_matrix',row)
        if file_hash(binding['path'])!=binding['sha256']:raise ValueError('original TSPLIB matrix changed: '+row['id'])
        raw=root/'data/raw/tsplib'/f'{row["id"]}.tsp'
        text=raw.read_text(encoding='utf-8')
        header,body=text.split('NODE_COORD_SECTION')
        fields={k.strip():v.strip() for line in header.splitlines() if ':' in line for k,v in [line.split(':',1)]}
        if fields['EDGE_WEIGHT_TYPE']!='EUC_2D':raise ValueError('unsupported original TSPLIB distance rule')
        coordinates={int(parts[0]):[float(parts[1]),float(parts[2])] for line in body.splitlines()
                     if (parts:=line.split()) and parts[0]!='EOF'}
        n=int(fields['DIMENSION'])
        if set(coordinates)!=set(range(1,n+1)):raise ValueError('incomplete TSPLIB coordinates')
        xy=np.asarray([coordinates[i] for i in range(1,n+1)])
        expected=np.floor(np.linalg.norm(xy[:,None]-xy[None,:],axis=2)+.5)
        with np.load(binding['path'],allow_pickle=False) as z:
            if not np.array_equal(z['distance_matrix'],expected):raise ValueError('original EUC_2D distance differs: '+row['id'])
        source=dict(row.get('source',{}),original_matrix=receipt(binding['path']),coordinates=receipt(raw),distance_rule=TSPLIB_DISTANCE_RULE)
        rows.append(dict(row,**receipt(binding['path']),source=source,evaluation_metric=TSPLIB_METRIC,reference_kind=TSPLIB_REFERENCE_RULE))
    result.update(test=rows,evaluation_metric=TSPLIB_METRIC,distance_rule=TSPLIB_DISTANCE_RULE,reference_rule=TSPLIB_REFERENCE_RULE)
    return result

def audit_instance_diversity(matrices):
    signatures=set()
    for matrix in matrices:
        a=np.asarray(matrix)
        signature=np.sort(a[np.triu_indices(len(a),1)]).round(10).tobytes()
        if signature in signatures:raise ValueError('duplicate original/supplement instance')
        signatures.add(signature)


def audit_training(task,rows,protocol=None):
    if protocol is not None:
        if task!='tsp' or protocol!=SUPPLEMENT_PROTOCOL or len(rows)!=128:
            raise ValueError('supplement protocol requires original64 + supplement64 TSP100')
        original=load_json(Path(__file__).resolve().parents[1]/'data/cef_v1_3_20261026/tsp_manifest.json')['train']
        if rows[:64]!=original: raise ValueError('original64 identities/references must remain unchanged')
        audit_training(task,rows[:64])
        audit_supplement(rows[64:])
        matrices=[]
        for row in rows:
            with np.load(row['path'],allow_pickle=False) as z: a=z['distance_matrix']
            matrices.append(a)
        audit_instance_diversity(matrices)
        ids=[r['id'] for r in rows]
        if len(set(ids))!=128: raise ValueError('duplicate training instance IDs')
        return dict(task=task,instances=128,original=64,supplement=64,valid=True,protocol=protocol)
    if len(rows)!=64: raise ValueError('64 training instances required')
    for r in rows:
        if r.get('split')!='train' or file_hash(r['path'])!=r['sha256']: raise ValueError('invalid training binding')
        with np.load(r['path'],allow_pickle=False) as z:
            a=z['distance_matrix' if task=='tsp' else 'processing_times']
        if not np.isfinite(a).all() or (a<0).any(): raise ValueError('invalid training values')
        if task=='tsp':
            if a.shape!=(100,100) or not np.allclose(a,a.T,rtol=0,atol=1e-12) or np.any(np.diag(a)!=0): raise ValueError('TSP100 symmetric matrix required')
            if r.get('reference_kind')!='Concorde_optimum' or not r.get('reference',0)>0: raise ValueError('matched Concorde reference required')
        elif a.shape[0]!=50 or not 2<=a.shape[1]<=20 or (a>=1).any(): raise ValueError('FSSP must be 50 jobs, 2..20 machines, [0,1)')
    return dict(task=task,instances=64,valid=True)


def audit_supplement(rows):
    from .tasks import TaskAdapter
    if len(rows)!=64: raise ValueError('64 supplemental instances required')
    for row in rows:
        if (row.get('split')!='train' or row.get('training_group')!='supplement' or
                row.get('reference_kind') not in ('HiGHS_numerically_bounded_tour','Concorde_numerically_bounded_unrounded_tour','SCIP_numerically_bounded_unrounded_tour')):
            raise ValueError('invalid supplemental reference/group binding')
        if file_hash(row['path'])!=row['sha256']: raise ValueError('supplement file changed')
        with np.load(row['path'],allow_pickle=False) as z:
            p=z['coordinates']; a=z['distance_matrix']
        if (p.shape!=(100,2) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or
                len(np.unique(p,axis=0))!=100 or a.shape!=(100,100) or
                not np.allclose(a,np.linalg.norm(p[:,None]-p[None,:],axis=2),rtol=0,atol=1e-12)):
            raise ValueError('invalid supplemental Euclidean coordinates/matrix')
        artifacts=[row['reference_record'],row['source']['generator'],row['source']['parameters']]
        for artifact in artifacts:
            if file_hash(artifact['path'])!=artifact['sha256']: raise ValueError('supplement provenance changed')
        ref=load_json(row['reference_record']['path'])
        if row['reference_kind'] in ('Concorde_numerically_bounded_unrounded_tour','SCIP_numerically_bounded_unrounded_tour') and ref.get('reference_kind')!=row['reference_kind']:
            raise ValueError('supplement native reference kind mismatch')
        objective=TaskAdapter('tsp').objective(dict(distance_matrix=a.tolist()),ref['solution'])
        lower=ref.get('lower_bound')
        if (not np.isfinite(row.get('reference',float('nan'))) or row['reference']<=0 or
                not np.isclose(objective,row['reference'],rtol=0,atol=1e-9) or
                not np.isclose(objective,ref['objective'],rtol=0,atol=1e-9) or
                isinstance(lower,bool) or not isinstance(lower,(float,int)) or not np.isfinite(lower) or
                lower<=0 or lower>objective+1e-7):
            raise ValueError('supplement reference tour/bound mismatch')
        gap=100*(objective/lower-1)
        if not np.isclose(gap,ref['bound_gap_pct'],atol=1e-7,rtol=0):
            raise ValueError('supplement reference numerical gap mismatch')
        if gap>row.get('reference_gap_tolerance_pct',0.00001)+1e-8:
            raise ValueError('supplement reference bound too weak')
        if type(row.get('generator_seed')) is not int or not 0<=row['generator_seed']<2**32:
            raise ValueError('supplement generator seed must be uint32')


def bind_supplement(base,path):
    """Explicitly extend a trusted original manifest; preserve its test bindings."""
    import copy
    supplement=load_json(path)
    if supplement.get('protocol')!=SUPPLEMENT_PROTOCOL or supplement.get('diagnostic') is not False:
        raise ValueError('formal frozen supplemental manifest required')
    result=copy.deepcopy(base)
    if len(result['train'])!=64: raise ValueError('supplement requires original64 base')
    result['train'].extend(supplement['train'])
    result.update(training_protocol=SUPPLEMENT_PROTOCOL,supplement_manifest=receipt(path))
    result['audit']=audit_training('tsp',result['train'],SUPPLEMENT_PROTOCOL)
    return result

def prepare(root,output,data_seed):
    """Import verified TSP and expose the unresolved author FSSP conflict."""
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    original=root/'problems/tsp_gls/TrainingData/TSPAEL64.pkl'
    # Trusted repository artifact only. Candidate code never gets pickle access.
    with original.open('rb') as f: data=pickle.load(f)
    if set(data)!=set(('coordinate','optimal_tour','distance_matrix','cost')) or any(len(v)!=64 for v in data.values()): raise ValueError('unexpected author TSP pickle schema')
    train=[]; folder=output/'tsp'; folder.mkdir()
    for i in range(64):
        coord=np.asarray(data['coordinate'][i]); matrix=np.asarray(data['distance_matrix'][i]); tour=np.asarray(data['optimal_tour'][i],dtype=int); ref=float(data['cost'][i])
        reconstructed=np.sqrt(((coord[:,None,:]-coord[None,:,:])**2).sum(axis=-1))
        if coord.shape!=(100,2) or (coord<0).any() or (coord>1).any() or not np.allclose(matrix,reconstructed,atol=1e-12,rtol=0): raise ValueError('author distance definition mismatch')
        if sorted(tour.tolist())!=list(range(100)) or not np.isclose(sum(matrix[tour[j],tour[(j+1)%100]] for j in range(100)),ref,atol=1e-9,rtol=0): raise ValueError('reference tour does not realize reference cost')
        p=folder/f'train_{i:04d}.npz'; np.savez(p,distance_matrix=matrix)
        train.append(dict(id=f'train_{i:04d}',split='train',**receipt(p),reference=ref,reference_kind='Concorde_optimum',source=receipt(original,kind='author published TSPAEL64.pkl; optimality is author-supplied, not re-certified here')))
    legacy=load_json(root/'data/bundles/v2/tsp_gls/validation.json'); byid={r['id']:r for r in legacy['instances']}
    test=[]
    for id in TSPLIB29:
        row=byid[id]; p=root/'data/bundles/v2/tsp_gls/instances'/f'{id}.npz'
        if file_hash(p)!=row['sha256']: raise ValueError('legacy test file differs: '+id)
        test.append(dict(id=id,split='test',**receipt(p),reference=row['reference'],reference_kind='published optimal tour',source=row['source']))
    test=bind_original_tsplib(dict(test=test))['test']
    tsp=dict(task='tsp',train=train,test=test,source_version=receipt(original),evaluation_metric=TSPLIB_METRIC,distance_rule='continuous Euclidean author TSP100; TSPLIB '+TSPLIB_DISTANCE_RULE,reference_rule=TSPLIB_REFERENCE_RULE,audit=audit_training('tsp',train))
    (output/'tsp_manifest.json').write_text(json.dumps(tsp,indent=2),encoding='utf-8')
    # Do not silently label the historical PCG64 U01 data as author-generator data.
    gen=root.parent/'examples/user_fssp_gls/TrainingData/instance_generator.py'
    source=gen.read_text(encoding='utf-8')
    if 'random.randint(1, 100)' not in source: raise ValueError('re-audit author generator: source changed')
    fssp=dict(task='fssp',status='blocked_author_generator_distribution_conflict',source_generator=receipt(gen),required_distribution='U[0,1)',observed_generator_distribution='integer 1..100',data_seed=data_seed,train=[],test=[],next_action='Supply a frozen author U01 generator or explicitly resolve the generator-versus-paper conflict; never use integer data for the required U01 training protocol.')
    (output/'fssp_data_blocker.json').write_text(json.dumps(fssp,indent=2),encoding='utf-8')
    return dict(tsp=str(output/'tsp_manifest.json'),fssp=fssp)

def prepare_fssp(root,output,data_seed):
    """Minimal explicit paper-distribution correction of the author generator.

    Main spec 09/10 require U01; preserve author Python randint machine rule.
    This is not claimed to reproduce the integer sample files byte-for-byte.
    """
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    original=root.parent/'examples/user_fssp_gls/TrainingData/instance_generator.py'
    text=original.read_text(encoding='utf-8')
    if text.count('random.randint(1, 100)')!=1: raise ValueError('unexpected author generator')
    corrected=text.replace('random.randint(1, 100)','random.uniform(0.0, 1.0)')
    (output/'author_generator_original.py').write_text(text,encoding='utf-8')
    (output/'author_generator_U01.py').write_text(corrected,encoding='utf-8')
    rng=random.Random(data_seed); rows=[]
    for i in range(64):
        m=rng.randint(2,20); p=np.array([[rng.uniform(0.0,1.0) for _ in range(m)] for _ in range(50)],dtype=float)
        path=output/f'train_{i:04d}.npz'; np.savez(path,processing_times=p)
        rows.append(dict(id=f'train_{i:04d}',split='train',**receipt(path),source=dict(original=receipt(original),corrected=receipt(output/'author_generator_U01.py'),data_seed=data_seed,generator='Python random.Random; exact author draw order, processing distribution changed explicitly by main specification')))
    tests=[]; references=[]; counter=0
    for n,m in SCALES:
        path=root.parent/f'examples/user_fssp_gls/TestingData/Taillard/t_j{n}_m{m}.txt'
        lines=path.read_text(encoding='utf-8-sig').splitlines(); i=0; count=0
        while i<len(lines):
            if 'number of jobs' not in lines[i].lower():
                if lines[i].strip(): raise ValueError('unexpected Taillard content')
                i+=1; continue
            header=list(map(int,lines[i+1].split()))
            if len(header)!=5 or header[:2]!=[n,m] or header[3]<=0: raise ValueError('invalid original header')
            if 'processing' not in lines[i+2].lower(): raise ValueError('processing marker missing')
            matrix=np.array([list(map(int,lines[i+3+k].split())) for k in range(m)],dtype=float).T
            if matrix.shape!=(n,m): raise ValueError('Taillard dimensions')
            counter+=1; count+=1; p=output/f'ta{counter:03d}.npz'; np.savez(p,processing_times=matrix)
            tests.append(dict(id=f'ta{counter:03d}',split='test',**receipt(p),reference=header[3],reference_kind='original Taillard header feasible upper bound',lower_bound=header[4],source=receipt(path)))
            i+=3+m
        if count!=10: raise ValueError('ten Taillard cases per scale required')
        references.append(receipt(path))
    manifest=dict(task='fssp',train=rows,test=tests,source_version=receipt(original),generator_correction=dict(author_file=receipt(original),frozen_corrected=receipt(output/'author_generator_U01.py'),change='randint(1,100) -> uniform(0,1); explicit authority:09 §16.1 and10 §2',identity_limit='paper-distribution adaptation, not byte-identical original integer example'),data_seed=data_seed,reference_files=references,audit=audit_training('fssp',rows))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def audit_test(task,rows):
    ids=[r['id'] for r in rows]
    expected=TSPLIB29 if task=='tsp' else [f'ta{i:03d}' for i in range(1,111)]
    if set(ids)!=set(expected) or len(ids)!=len(expected): raise ValueError('complete prescribed benchmark required')
    if task=='tsp':
        metrics={r.get('evaluation_metric') for r in rows}
        if len(metrics)!=1 or not metrics<={TSPLIB_METRIC,UNROUNDED_TSPLIB_METRIC}:
            raise ValueError('unlabelled or mixed TSPLIB evaluation metrics; bind original distances for a new run')
    for r in rows:
        if r.get('split')!='test' or file_hash(r['path'])!=r['sha256'] or r.get('reference',0)<=0: raise ValueError('invalid test/reference binding')
    return dict(task=task,count=len(rows),valid=True)
