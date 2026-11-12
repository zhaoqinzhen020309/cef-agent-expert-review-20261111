"""先准备固定补充64题，再运行runCEFWithData.py进行正式算法搜索。"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True

CONFIG=dict(
    model='deepseek-flash',
    base_url='https://api.deepseek.com',
    thinking=False,  # 数据生成使用标准非思考调用，正文预算不与显式推理共享。
    reasoning_effort='low',  # 仅在thinking=True时发送。
    max_output_tokens=8192,
    iterations=2,  # 数据设计轮数；每轮提交并实际评估一个生成程序。
    data_seed=2025,
    solver_seed=2025,
    workers=4,
    instances=64,
    n_cities=100,
    solver_seconds=60,
    generator_seconds=5,
    reference_seconds=120,
    reference_gap_pct=0.00001,  # 数值上下界允许的最大百分比差。
    # 按原训练成绩选出的固定校准程序；不把其TSPLIB结果输入Agent。
    solver_source=ROOT/'experiments/pab_common_start/tsp_s2025_20261031_003524_554089/search/best_solver.py',
    output_dir=ROOT/'data/agent_supplement/tsp100_seed2025',
    console_progress=True,
)

if __name__=='__main__':
    from scripts.prepare_cef_training import main
    raise SystemExit(main(CONFIG))
