"""默认手动逐代：完成一代后暂停，使用--evaluate评估，--next继续。"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True

CONFIG=dict(
    model='deepseek-flash',
    base_url='https://api.deepseek.com',
    rounds=None,  # 每次运行一代后暂停；--next继续下一代。
    tsplib_metric='tsplib_unrounded_original_reference_v1',
    round_benchmark=True,
    evaluate_each_round=False,  # 训练后暂停；手动--evaluate查看TSPLIB。
    baseline_benchmark=ROOT/'experiments/cef_manual_rounds/tsp_s2025_20261110_004744_291083/starting_solver_tsplib29_unrounded_20261111_102715_190307/summary.json',
    seeds=[2025],
    data_seed=2025,
    workers=8,
    integrated_feedback_v1=True,
    agent_tool_loop_v1=True,
    agent_task_runtime_v2=True,
    measured_research_loop_v1=True,
    exploration_efficiency_v1=True,
    tool_context_window_tokens=1_000_000,  # DeepSeek官方公布的上下文容量；不是响应次数或搜索停止条件。
    tool_max_turns=None,
    tool_timeout_s=10,
    tool_max_output_tokens=32768,
    reference_workers=1,  # 参考计算串行，降低本机内存峰值；算法评估仍8并行。
    data_thinking=False,
    data_max_output_tokens=32768,
    # 明确热启动；初始算法和既有数据经验均保存来源，不冒充冷启动。
    initial_solver=ROOT/'experiments/pab_common_start/tsp_s2025_20261031_003524_554089/search/best_solver.py',
    initial_supplement=ROOT/'data/agent_supplement/tsp100_seed2025/supplement_manifest.json',
    common_start_probe=True,
    probe_states=8,
    probe_seconds=3.,
    output_dir=ROOT/'experiments/cef_focused_tasks',
)

if __name__=='__main__':
    from cef_log_output import launch
    raise SystemExit(launch(CONFIG))
