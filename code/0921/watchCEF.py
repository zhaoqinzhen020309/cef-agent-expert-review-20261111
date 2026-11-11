"""Read-only live view of CEF journals. Never imports repairing Archive readers."""
import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parent
KINDS={'model_started','tool_started','tool_completed','experimental_start','experimental_execution',
       'experimental_complete','data_round_start','round_complete','paused','blocked','failure','submitted','context_restarted','context_pruned','context_pressure'}
TOKENS=[('"kind": "'+kind+'"').encode() for kind in KINDS]


def short(value,limit=150):
    text=re.sub(r'[\x00-\x1f\x7f]+',' ',str(value)).strip()
    return text if len(text)<=limit else text[:limit]+'…'


class JournalTail:
    def __init__(self,path):self.path=Path(path);self.offset=0
    def poll(self):
        rows=[]
        try:
            with self.path.open('rb') as stream:
                if self.path.stat().st_size<self.offset:self.offset=0
                stream.seek(self.offset)
                while True:
                    line=stream.readline()
                    if not line or not line.endswith(b'\n'):break
                    self.offset=stream.tell()
                    if not any(token in line for token in TOKENS):continue
                    try:record=json.loads(line)
                    except ValueError:continue
                    if record.get('kind') in KINDS:rows.append(record)
        except FileNotFoundError:pass
        return rows


class LogView:
    def __init__(self,root):
        self.root=Path(root);self.experiments={};self.round=None;self.last_activity='尚无行动记录';self.activity_time=None
    def actor(self,path):
        path=Path(path)
        name=str(path).replace('\\','/')
        if '/data/agent_task/' in name:return '数据Agent'
        if 'editor' in name:return '算法编辑Agent'
        if 'planner' in name:return '算法规划Agent'
        return '流程'
    def consume(self,path,record):
        data=record['data'];kind=record['kind'];actor=self.actor(path)
        stamp=datetime.fromtimestamp(record.get('time',time.time())).strftime('%H:%M:%S')
        prefix=f'{stamp} | '+(f'第{self.round}轮 | ' if self.round is not None else '')+actor
        if kind=='model_started':
            self.last_activity=f'{actor}第{data["turn"]}次响应，等待模型返回';self.activity_time=record.get('time')
            return [f'{prefix} | 第{data["turn"]}次响应：等待模型返回']
        if kind=='tool_started':
            function=data['call'].get('function') or {};name=function.get('name','未知工具')
            try:args=json.loads(function.get('arguments','{}'))
            except (ValueError,TypeError):args={}
            if name=='read_artifact':action='读取 '+short(args.get('artifact_id',''))+f'（起点{args.get("offset",0)}）'
            elif name=='write_file':action='保存 '+short(args.get('relative_path',''))+f'（{len(args.get("content",""))}字符）'
            elif name=='run_python':action='执行Python检查 '+short(args.get('script_path','临时检查脚本'))+' | 输入：'+short(', '.join(args.get('input_artifact_ids',[])) or '无外部输入')
            elif name=='run_solver_experiment':action='启动求解实验 | 程序：'+short(args.get('source_id',''))
            elif name=='measure_data_candidate':action='测量候选出题规则 '+short(args.get('candidate_path',''))+' | 固定题目索引：'+short(str(args.get('case_indices',[])))
            elif name=='run_experiment_batch':action='执行实验计划 '+short(args.get('question',''))+f' | {len(args.get("jobs",[]))}项实验'
            elif name=='submit_candidate':action='提交方案 '+short(args.get('path','内联方案'))+'，等待程序校验'
            elif name=='checkpoint_context':action='保存探索进度并整理上下文'
            elif name=='list_artifacts':action='查看可用材料与文件索引'
            elif name=='report_blocker':action='报告阻断：'+short(args.get('reason',''))
            else:action='调用 '+short(name)
            self.last_activity=actor+'：'+action;self.activity_time=record.get('time')
            return [f'{prefix} | 第{data["turn"]}次响应：{action}']
        if kind=='tool_completed':
            name=(data['call'].get('function') or {}).get('name');result=data['result']
            self.last_activity=actor+'工具已返回';self.activity_time=record.get('time')
            if result.get('not_executed'):return [f'{prefix} | 任务目标已验证，此后续调用未执行']
            if name=='submit_candidate':
                return [f'{prefix} | '+('提交校验通过，交付完成' if result.get('accepted') else '提交校验未通过：'+short(result.get('error','原因见原始记录')))]
            if result.get('error'):return [f'{prefix} | 工具执行失败：'+short(result['error'])]
            if result.get('task_completion',{}).get('accepted'):
                return [f'{prefix} | 完整出题计划已验证改善，交付校验通过；本轮数据研究完成']
            if name=='run_experiment_batch':
                rows=result.get('experiments',[])
                return [f'{prefix} | 实验计划完成：{len(rows)}项，工具失败{sum(not r["result"].get("ok",False) for r in rows)}项 | 证据：'+short(result.get('artifact_id',''))]
            if name=='measure_data_candidate':
                gap=result.get('mean_reference_gap_pct')
                score=f'已测子集平均gap={gap:.6f}%' if gap is not None else '尚无可汇总参考gap'
                return [f'{prefix} | 候选题测量返回：完成{result.get("measured_case_count",0)}/{len(result.get("requested_case_indices",[]))} | '+score+
                    f' | 参考待完成{len(result.get("pending_cases",[]))}，失败{len(result.get("failed_cases",[]))} | 证据：'+short(result.get('artifact_id','见工具记录'))]
            if name=='run_python':
                seconds=result.get('wallclock_s',result.get('elapsed_s'))
                output=result.get('result')
                if output is None:output=result.get('stdout','').strip()
                suffix=(' | 输出：'+short(json.dumps(output,ensure_ascii=False) if isinstance(output,(dict,list)) else output)) if output not in (None,'') else ''
                return [f'{prefix} | 检查执行成功'+(f'，用时{seconds:.2f}s' if seconds is not None else '')+suffix]
            return []
        if kind=='experimental_start':
            try:info=json.loads(Path(data['artifact_path']).read_text(encoding='utf-8'))
            except (OSError,ValueError):info={}
            spec=info.get('input_spec',{});conditions=info.get('conditions',{})
            total=len(spec.get('instance_ids') or spec.get('seeds') or info.get('input_cases',[]))
            entry=dict(total=total,valid=0,failed=0,gaps=[],actor=self.actor(data['artifact_path']))
            self.experiments[data['experiment_id']]=entry
            seconds=conditions.get('solver_time_limit_s');formal=conditions.get('formal_solver_time_limit_s')
            mode='正式时限' if seconds==formal else f'诊断；正式时限{formal}s'
            target=('探题生成器 '+short(spec['generator_id'])) if spec.get('generator_id') else '指定训练题'
            return [f'{stamp} | {entry["actor"]} | 实验{data["experiment_id"][:8]}：'+
                    f'{short(info.get("source_id","程序"))} → {target}，共{total}题，每题上限{seconds}s（{mode}）']
        if kind=='experimental_execution':
            entry=self.experiments.setdefault(data['experiment_id'],dict(total=0,valid=0,failed=0,gaps=[],actor='实验'))
            entry['valid' if data['valid'] else 'failed']+=1
            if data['valid'] and data.get('reference_gap_pct') is not None:entry['gaps'].append(data['reference_gap_pct'])
            done=entry['valid']+entry['failed'];total=entry['total'] or '?'
            error=(' | '+short(data.get('error'))) if not data['valid'] else ''
            return [f'{stamp} | {entry["actor"]} | 实验{data["experiment_id"][:8]} {done}/{total}：合法{entry["valid"]}，失败{entry["failed"]}'+error]
        if kind=='experimental_complete':
            entry=self.experiments.get(data['experiment_id'])
            if not entry:return []
            if data.get('status')=='generation_failed':
                return [f'{stamp} | {entry["actor"]} | 实验{data["experiment_id"][:8]}：生成阶段失败，未开始求解']
            done=entry['valid']+entry['failed']
            if done and len(entry['gaps'])==done:score=f'平均gap={sum(entry["gaps"])/done:.6f}%'
            elif entry['failed']:score='存在失败，不汇总平均gap'
            else:score='无参考gap，仅记录实际目标值'
            return [f'{stamp} | {entry["actor"]} | 实验{data["experiment_id"][:8]}完成：合法{entry["valid"]}/{done} | {score}']
        if kind=='data_round_start':
            self.round=data['round'];return [f'{stamp} | 第{self.round}轮：数据设计开始']
        if kind=='round_complete':return [f'{stamp} | 第{data["round"]}轮完成 | 当前训练gap={data["winner"]["fitness"]:.6f}%']
        if kind in ('paused','blocked','failure'):
            labels=dict(paused='已暂停',blocked='任务阻断',failure='运行失败')
            return [f'{prefix} | {labels[kind]}：'+short(data.get('reason',data.get('error','见原始记录')))]
        if kind=='context_restarted':return [f'{prefix} | 已保存进度并继续原任务']
        if kind=='context_pruned':return [f'{prefix} | 已将{len(data["results"])}条旧工具结果收纳为可读取证据，上下文保留摘要与引用']
        if kind=='context_pressure':return [f'{prefix} | 上下文接近模型容量，已请求保存研究进度；搜索任务仍继续']
        return []


def journals(root):
    return [root/'search/facts.jsonl',*root.glob('rounds/*/data/agent_task/session/facts.jsonl'),
            *root.glob('search/agent_tasks/*/session/facts.jsonl')]


def main(argv=None):
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    parser=argparse.ArgumentParser(description='只读查看CEF当前行动、工具结果及实验条件，不调用模型。')
    parser.add_argument('run',nargs='?',type=Path,help='运行目录；不填则选择最近的runtime_v2实验')
    parser.add_argument('--once',action='store_true',help='显示最近记录后退出')
    parser.add_argument('--lines',type=int,default=25)
    args=parser.parse_args(argv)
    root=args.run
    if root is None:
        candidates=list((ROOT/'experiments/cef_runtime_v2').glob('*/search_config.json'))
        if not candidates:parser.error('没有找到runtime_v2实验，请指定完整运行目录')
        root=max(candidates,key=lambda p:p.parent.stat().st_mtime).parent
    root=root.resolve()
    if not (root/'search_config.json').is_file():parser.error('指定目录没有search_config.json')
    if args.lines<1:parser.error('--lines必须为正整数')
    print('只读实时日志：'+str(root),flush=True)
    print('显示工具行动和真实结果；探索实验不作为正式训练或TSPLIB成绩。Ctrl+C只退出本窗口。',flush=True)
    view=LogView(root);tails={};first=True
    last_output=time.monotonic()
    try:
        while True:
            records=[]
            for path in journals(root):
                if path not in tails:tails[path]=JournalTail(path)
                records.extend((record,path) for record in tails[path].poll())
            records.sort(key=lambda item:(item[0].get('time',0),str(item[1]),item[0].get('seq',0)))
            lines=[]
            for record,path in records:lines.extend(view.consume(path,record))
            if first:lines=list(deque(lines,maxlen=args.lines));first=False
            for line in lines:print(line,flush=True)
            if lines:last_output=time.monotonic()
            elif not args.once and time.monotonic()-last_output>=30:
                age=max(0,int(time.time()-view.activity_time)) if view.activity_time is not None else None
                print('暂无新日志；最后记录：'+view.last_activity+(f'（距今{age}秒）' if age is not None else ''),flush=True)
                last_output=time.monotonic()
            if args.once:return 0
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n日志查看已退出，实验继续运行。',flush=True)
        return 0


if __name__=='__main__':raise SystemExit(main())
