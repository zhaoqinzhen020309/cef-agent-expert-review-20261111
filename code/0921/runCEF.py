"""从原始程序开始完整PAB迭代，用共同解起点实验辅助后续修改。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts.cef_launcher import run_pab as run

CONFIG = dict(
    model="deepseek-flash",
    base_url="https://api.deepseek.com",

    # 每个种子从原始程序重新生成算法，彼此不继承代码、反馈或模型会话。
    # 种子控制求解器随机行为；DeepSeek本身不支持固定采样seed。
    seeds=[2026],

    # 每次实验生成并完整评估10个候选；诊断动作不增加迭代数。
    iterations=10,

    # 同时评估的求解器进程数；每个进程只运行一个实例。
    workers=4,

    # 共同解起点配对短跑；每状态两个预登记重启种子。
    common_start_probe=True,
    probe_states=8,
    probe_seconds=3.0,

    # 每个种子单独保存新生成的代码、模型调用、训练和测试结果。
    output_dir=ROOT / "experiments" / "pab_common_start",
)

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
