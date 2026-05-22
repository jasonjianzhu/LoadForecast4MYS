# 实验清单

完整结论见 [experiment_conclusion.md](./experiment_conclusion.md)。

## 保留实验（PatchTST）

| 代号 | 配置 | 输出 |
|------|------|------|
| E04 基线 | `configs/experiments/patchtst_e04_balanced_sampler.json` | `outputs/experiments/patchtst_e04_balanced_sampler/` |
| E04+E02（推荐） | `configs/experiments/patchtst_e04_e02_peak_loss.json` | `outputs/experiments/patchtst_e04_e02_peak_loss/` |

## 运行

```bash
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_balanced_sampler.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_e02_peak_loss.json
```
