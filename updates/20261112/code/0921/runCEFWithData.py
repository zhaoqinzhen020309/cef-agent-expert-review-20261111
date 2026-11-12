"""使用已冻结的补充64题，复用runCEF的完整PAB搜索配置。"""
from runCEF import CONFIG as BASE_CONFIG, ROOT
from scripts.cef_launcher import run_pab

CONFIG=dict(BASE_CONFIG,
    supplement_manifest=ROOT/'data/agent_supplement/tsp100_seed2025/supplement_manifest.json',
    output_dir=ROOT/'experiments/pab_with_agent_data')

if __name__=='__main__':
    raise SystemExit(run_pab(CONFIG))
