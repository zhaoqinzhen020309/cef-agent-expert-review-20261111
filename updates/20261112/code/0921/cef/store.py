"""Append-only, fsync'd facts; interpretations are separate revision events."""
import json
import os
import time
import threading
from pathlib import Path
from .patch import digest


def canonical(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,allow_nan=False)


class Archive:
    def __init__(self,root):
        self.lock=threading.RLock()
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'facts.jsonl'
        self.records=[]
        if self.path.exists():
            raw=self.path.read_bytes(); lines=raw.splitlines(keepends=True)
            for i,line in enumerate(lines):
                try: item=json.loads(line)
                except (ValueError,UnicodeDecodeError):
                    # Preserve the torn tail, then repair only that uncommitted append.
                    if i!=len(lines)-1: raise ValueError('corrupt fact journal')
                    (self.root/f'torn_tail_{time.time_ns()}.bin').write_bytes(line)
                    with self.path.open('r+b') as f: f.truncate(sum(map(len,lines[:i])))
                    break
                if item['seq']!=len(self.records): raise ValueError('journal sequence mismatch')
                self.records.append(item)

    def add(self,kind,data):
        with self.lock:
            item=dict(seq=len(self.records),kind=kind,time=time.time(),data=data)
            with self.path.open('a',encoding='utf-8',newline='\n') as f:
                f.write(canonical(item)+'\n'); f.flush(); os.fsync(f.fileno())
            self.records.append(item)
            return item

    def of(self,kind):
        with self.lock: return [r['data'] for r in self.records if r['kind']==kind]

    def source(self,text):
        h=digest(text); p=self.root/'sources'/f'{h}.py'; p.parent.mkdir(exist_ok=True)
        if p.exists():
            with p.open(encoding='utf-8',newline='') as f: existing=f.read()
            if existing!=text: raise ValueError('source corruption')
        else:
            with p.open('x',encoding='utf-8',newline='') as f: f.write(text)
        return h

    def read_source(self,h):
        with (self.root/'sources'/f'{h}.py').open(encoding='utf-8',newline='') as f: text=f.read()
        if digest(text)!=h: raise ValueError('source hash mismatch')
        return text

    def artifact(self,category,value):
        """Immutable JSON payload; the journal retains its digest and relative path."""
        if category not in ('snapshots','probe_manifests','probe_results'):
            raise ValueError('unknown research artifact category')
        text=canonical(value); h=digest(text)
        path=self.root/category/(h+'.json'); path.parent.mkdir(exist_ok=True)
        if path.exists():
            if path.read_text(encoding='utf-8')!=text: raise ValueError('research artifact corruption')
        else:
            with path.open('x',encoding='utf-8',newline='\n') as stream:
                stream.write(text); stream.flush(); os.fsync(stream.fileno())
        return dict(sha256=h,path=path.relative_to(self.root).as_posix())


class ExperienceStore:
    def __init__(self,archive): self.archive=archive
    def create(self,card_id,facts):
        if any(c['card_id']==card_id for c in self.archive.of('card')): raise ValueError('duplicate card')
        self.archive.add('card',dict(card_id=card_id,**facts))
    def revise(self,card_id,interpretation,status,evidence_ids):
        if status not in ('待复查','本次观察支持','出现不一致'): raise ValueError('invalid experience status')
        if not any(c['card_id']==card_id for c in self.archive.of('card')): raise ValueError('unknown card')
        self.archive.add('card_revision',dict(card_id=card_id,interpretation=interpretation,status=status,evidence_ids=evidence_ids))
    def cards(self):
        cards=[]
        for c in self.archive.of('card'):
            c=dict(c,revision_history=[r for r in self.archive.of('card_revision') if r['card_id']==c['card_id']])
            c.update(status='待复查',interpretation='')
            if c['revision_history']: c.update({k:c['revision_history'][-1][k] for k in ('status','interpretation')})
            cards.append(c)
        return cards
    def retrieve(self,parent_hash,ancestors,functions):
        cards=self.cards()
        def key(pair):
            i,c=pair
            rank=0 if c['parent_hash']==parent_hash else 1 if c['parent_hash'] in ancestors[:2] else 2 if set(c.get('affected_functions',[]))&set(functions) else 3
            return rank,0 if c['status'] in ('待复查','出现不一致') else 1,-i
        return [c for _,c in sorted(enumerate(cards),key=key)[:3]]
