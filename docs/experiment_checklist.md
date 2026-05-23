# 实验清单

完整结论见 [experiment_conclusion.md](./experiment_conclusion.md)。

## 保留实验（PatchTST）

| 代号 | 配置 | 输出 |
|------|------|------|
| E04 基线 | `configs/experiments/patchtst_e04_balanced_sampler.json` | `outputs/experiments/patchtst_e04_balanced_sampler/` |
| E04+E02（单模型推荐） | `configs/experiments/patchtst_e04_e02_peak_loss.json` | `outputs/experiments/patchtst_e04_e02_peak_loss/` |
| Routed（推荐上线） | `scripts/build_routed_experiment.py` | `outputs/experiments/patchtst_station_routed/` |

## 运行

```bash
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_balanced_sampler.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_e02_peak_loss.json
./.venv/bin/python scripts/build_routed_experiment.py
```
