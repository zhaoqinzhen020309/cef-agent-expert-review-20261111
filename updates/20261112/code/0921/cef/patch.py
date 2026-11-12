"""Exact edits against one immutable full-program source."""
import ast
import hashlib


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def check_source(text, require_research=False):
    tree = ast.parse(text)
    entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve']
    if len(entries) != 1 or [a.arg for a in entries[0].args.args] != ['instance','time_limit_s','seed']:
        raise ValueError('required solve(instance, time_limit_s, seed) entry')
    if require_research:
        args=entries[0].args
        optional={a.arg:default for a,default in zip(args.kwonlyargs,args.kw_defaults)}
        if (args.posonlyargs or args.vararg or args.kwarg or args.defaults or
            set(optional)!={'initial_solution','observer'} or
            any(not isinstance(default,ast.Constant) or default.value is not None
                for default in optional.values())):
            raise ValueError('research entry requires keyword-only initial_solution=None, observer=None')
    return tree


def research_capable(source):
    """Whether a source explicitly implements the optional research entry."""
    try:
        check_source(source,require_research=True)
    except (SyntaxError,ValueError,TypeError):
        return False
    return True


def compose(parent, package):
    if digest(parent) != package['parent_hash']:
        raise ValueError('parent hash mismatch')
    edits = package['edits']
    if len(edits) not in (1,2): raise ValueError('one or two edits required')
    if not package.get('single_edit_reason' if len(edits)==1 else 'joint_reason'):
        raise ValueError('edit grouping reason required')
    spans = []
    for i,e in enumerate(edits):
        for key in ('edit_id','purpose','prediction','exact_replacements'):
            if not e.get(key): raise ValueError('missing edit '+key)
        for r in e['exact_replacements']:
            old,new = r['old'],r['new']
            if not old or parent.count(old) != 1: raise ValueError('anchor must occur exactly once')
            start=parent.index(old); end=start+len(old)
            if any(start<b and a<end for a,b,_,_ in spans): raise ValueError('overlapping edits')
            spans.append((start,end,new,i))
    result={'P':parent}
    subsets=[('PA',{0})] if len(edits)==1 else [('PA',{0}),('PB',{1}),('PAB',{0,1})]
    for label,indices in subsets:
        text=parent
        for a,b,new,i in sorted(spans,reverse=True):
            if i in indices: text=text[:a]+new+text[b:]
        # Syntax/entry validity is recorded per variant, never prune a joint
        # because either component is invalid in isolation.
        result[label]=text
    return result
