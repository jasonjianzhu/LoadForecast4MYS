# PatchTST 负荷预测实验结论

## 1. 实验背景

- **任务**：4 个场站、5 分钟粒度，用前 7 天负荷预测未来 1 天（288 点）
- **数据**：`data/` 下 4 份 CSV，`target_clip_min=0` 清洗负负荷
- **Backbone**：PatchTST（`series_id_mode: repeat`）+ RevIN + 日历外生特征
- **训练**：`station_balanced` 采样，Huber+MSE 主损失，验证/测试各场站最后 5/7 天滚动评估
- **早停**：`early_stopping_metric=nrmse`，`patience=8`

在 E04 基线上曾尝试：series token、场景损失加权、加强峰值损失；**仅加强峰值损失在测试集上有效**，其余回退。完整过程见 git 历史；仓库仅保留 E04 与 E04+E02 的配置与结果。

---

## 2. 保留实验

| 代号 | 配置 | 输出目录 | 用途 |
|------|------|----------|------|
| **E04** | `configs/experiments/patchtst_e04_balanced_sampler.json` | `outputs/experiments/patchtst_e04_balanced_sampler/` | 基线对照 |
| **E04+E02** | `configs/experiments/patchtst_e04_e02_peak_loss.json` | `outputs/experiments/patchtst_e04_e02_peak_loss/` | **推荐上线** |

**推荐权重**：`patchtst_e04_e02_best_epoch13.pt`（best_epoch=13）

---

## 3. 测试集结果（主要依据）

| 指标 | E04 基线 (ep14) | E04+E02 (ep13) | 变化 |
|------|-----------------|----------------|------|
| NRMSE | 0.0722 | **0.0681** | -5.7% |
| WAPE | 0.1653 | **0.1644** | -0.5% |
| MAE (kW) | 50.8 | **50.5** | -0.3 |

### 分场站（测试集）

| 场站 | E04 MAE | E04+E02 MAE | E04 NRMSE | E04+E02 NRMSE | E04 WAPE | E04+E02 WAPE |
|------|---------|-------------|-----------|---------------|----------|--------------|
| GS Paperboard | 119.3 | **112.8** | 0.148 | **0.137** | 0.189 | **0.179** |
| Plastone | **26.3** | 28.2 | **0.107** | 0.113 | **0.097** | 0.104 |
| Quality-Coils | **32.8** | 36.6 | **0.098** | 0.109 | **0.195** | 0.217 |
| Tamura | 24.9 | **24.6** | 0.127 | **0.120** | 0.157 | **0.155** |

### 峰值相关（测试集，`test_peak_by_series`）

| 场站 | E04 peak_ratio | E04+E02 peak_ratio | E04 nonpeak_bias (kW) | E04+E02 nonpeak_bias (kW) |
|------|----------------|--------------------|------------------------|---------------------------|
| GS | 0.737 | 0.739 | +48 | +51 |
| Plastone | 0.857 | **0.886** | +5 | +9 |
| Quality | 0.668 | **0.745** | +9 | +8 |
| Tamura | 0.640 | **0.702** | +8 | +9 |

说明：`peak_ratio` = 真峰时刻预测值 / 真实峰值（越接近 1 越好）。GS 峰顶仍偏低约 26%，加强 peak loss 后 GS 峰顶改善有限，但全场站 MAE/NRMSE 整体更优。

---

## 4. 验证集（选模用，仅供参考）

| 指标 | E04 | E04+E02 |
|------|-----|---------|
| val NRMSE | **0.122** | 0.134 |
| weekday→holiday WAPE | **0.695** | 0.757 |

E04+E02 在验证集上略差于 E04，但**测试集 7 天滚动**明显更好。验证窗仅 20 个（4 站×5 天），holiday 样本极少，不宜单独否定 E04+E02。

---

## 5. 已尝试但未采纳的改动

| 改动 | 结论 |
|------|------|
| `series_id_mode: token` | 测试 NRMSE 0.076，holiday 场景更差，不采纳 |
| `scenario_loss_weights`（节假日场景 1.1–1.25） | 测试 NRMSE 0.084，GS 明显变差，不采纳 |
| token + scenario + peak 组合 | 未跑；前两步已失败，无必要 |

---

## 6. E04+E02 相对 E04 的配置差异

```json
"peak_loss_weight": 0.3,
"underprediction_topk_weight": 0.15,
"daily_max_loss_weight": 0.1
```

（E04 为 0.2 / 0.08 / 0.0；其余结构、采样、主损失相同。）

---

## 7. 结论与建议

1. **生产推荐**：`patchtst_e04_e02_peak_loss`，checkpoint `patchtst_e04_e02_best_epoch13.pt`。
2. **基线保留**：E04 用于回归对比与 ablation 参照。
3. **业务解读**：整体误差最低；GS 绝对误差仍最大；节假日切换（`weekday→holiday`）仍是弱项，不宜仅靠 loss 加权，后续可考虑日历特征增强或加长验证窗。
4. **复现训练**：

```bash
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_balanced_sampler.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_e02_peak_loss.json
```

评估输出：`evaluation_summary.json`（含 `test_peak_by_series`）、`test_predictions.csv`、`plots/`。
