# 实验清单

## 原则

- 保持同一份原始数据：`data/`
- 保持同一份时序切分：最后 `5` 天验证、最后 `7` 天测试
- 保持同一套评估指标：`MAE / RMSE / NMAE / NRMSE / WAPE / MAPE`
- 重点观察：
  - `GS / Tamura / Quality` 的高峰段误差
  - `pred_at_true_peak`
  - `Plastone` 的 `nonpeak bias`

## 实验矩阵

### `E00` `TimeXer` `0.2.0` 基线复跑

- 配置：`configs/experiments/timexer_e00_v020_baseline.json`
- 目的：用统一代码框架复现稳定基线，作为后续所有实验的对照
- 关键设置：
  - `patch_len=12`
  - `patch_stride=12`
  - `series_id_embedding_dim=2`
  - `0.5 * Huber + 0.5 * MSE`
  - pointwise peak focus
  - no target clipping
  - no holiday/scenario weighting
  - no station loss weighting

### `E01` 仅改负负荷清洗

- 配置：`configs/experiments/timexer_e01_target_clip.json`
- 目的：验证将训练目标裁剪到 `load >= 0` 是否能改善 `Plastone / Quality`
- 相对 `E00` 唯一变化：
  - `target_clip_min = 0.0`

### `E02` 仅改峰值辅助损失

- 配置：`configs/experiments/timexer_e02_peak_loss.json`
- 目的：验证“稳态主损失 + 轻量峰值辅助”是否能改善 `GS / Tamura / Quality`
- 相对 `E01` 主要变化：
  - `L_base = 0.7 * Huber + 0.3 * MSE`
  - `pointwise peak focus` 关闭
  - `top-k peak loss` 打开
  - `underprediction on top-k` 打开
  - `daily max loss` 关闭

### `E03` 仅改节假日场景加权

- 目的：验证节假日切换场景的轻量加权是否有收益
- 建议场景权重：
  - `weekday->holiday = 1.25`
  - `holiday->weekday = 1.20`
  - `holiday->holiday = 1.15`
  - `holiday->weekend = 1.10`
  - 其他 = `1.00`

### `E04` 仅改站点采样

- 配置：`configs/experiments/timexer_e04_balanced_sampler.json`
- 目的：增加 `Plastone` 曝光，但避免整站 loss 放大
- 建议方式：
  - 站点均衡 batch
  - 或 `WeightedRandomSampler`
- 说明：
  - 优先改采样，不优先改 loss 权重

### `E05` 仅改 `series_id` 表达

- 配置：`configs/experiments/timexer_e05_series_token.json`
- 目的：减少站点身份被误当作时变特征
- 建议变化：
  - 从“平铺到每个时间步”改为“独立静态 token”

### `E06` 仅改 patch 设计

- 目的：单独验证重叠 patch 对峰值的影响
- 对比方案：
  - `patch_len=12, patch_stride=12`
  - `patch_len=12, patch_stride=6`

### `E07` `TimeXer` 最优组合

- 从 `E01-E06` 中选择 2 到 3 个最有效改动组合
- 要求：
  - 总体指标不劣于 `E00`
  - `GS / Tamura / Quality` 高峰段误差下降
  - `Plastone` 不出现明显基线抬高

### `E08` `ModernTCN`

- 配置：`configs/experiments/moderntcn_e04_balanced_sampler.json`
- 目的：验证卷积型 backbone 对 `5min` 高频工业负荷是否更友好
- 约束：
  - 保持同样数据切分
  - 保持相同日历协变量
  - 尽量保持与 `E07` 相同的损失设计
  - 第一版直接与 `E04` 组合，使用站点均衡采样

### `E09` `PatchTST`

- 配置：`configs/experiments/patchtst_e04_balanced_sampler.json`
- 目的：验证长 lookback patch backbone 的上限
- 约束：
  - 同 `E08`

### `E10` `iTransformer`

- 目的：作为第三个 backbone 备选做横向对比
- 优先级：
  - 低于 `ModernTCN` 和 `PatchTST`

## 建议顺序

1. `E00`
2. `E01`
3. `E02`
4. `E04`
5. `E05`
6. `E06`
7. `E07`
8. `E08`
9. `E09`

## 运行命令

```bash
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/timexer_e00_v020_baseline.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/timexer_e01_target_clip.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/timexer_e02_peak_loss.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/timexer_e04_balanced_sampler.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/timexer_e05_series_token.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/moderntcn_e04_balanced_sampler.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_balanced_sampler.json
```
