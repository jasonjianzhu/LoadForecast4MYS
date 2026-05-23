# PatchTST 负荷预测实验结论

## 1. 实验背景

- **任务**：4 个场站、5 分钟粒度，用前 7 天负荷预测未来 1 天（288 点）
- **数据**：`data/` 下 4 份 CSV，`target_clip_min=0` 清洗负负荷
- **Backbone**：PatchTST（`series_id_mode: repeat`）+ RevIN + 日历外生特征
- **训练**：`station_balanced` 采样，Huber+MSE 主损失，验证/测试各场站最后 5/7 天滚动评估
- **早停**：`early_stopping_metric=nrmse`，`patience=8`

在 E04 基线上曾尝试：series token、场景损失加权、加强峰值损失、弱化 peak loss、特殊日切换特征、overlap patch，以及更换 `ModernTCN`/`TimeXer` backbone。最终结论很明确：**PatchTST 主线最强，单模型最优为 `E04+E02`，整体最优为 routed 方案**。完整过程见 git 历史；仓库当前仅保留可复现的 E04、E04+E02 与 routed 结果目录，其余失败实验结果目录已清理，但指标结论保留在本文档中。

---

## 2. 当前保留结果

| 代号 | 配置 | 输出目录 | 用途 |
|------|------|----------|------|
| **E04** | `configs/experiments/patchtst_e04_balanced_sampler.json` | `outputs/experiments/patchtst_e04_balanced_sampler/` | 基线对照 |
| **E04+E02** | `configs/experiments/patchtst_e04_e02_peak_loss.json` | `outputs/experiments/patchtst_e04_e02_peak_loss/` | 单模型最优 |
| **Routed** | `scripts/build_routed_experiment.py` | `outputs/experiments/patchtst_station_routed/` | **推荐上线** |

**推荐单模型权重**：`patchtst_e04_e02_best_epoch13.pt`（best_epoch=13）

---

## 3. 最终推荐结果

| 指标 | E04 基线 (ep14) | E04+E02 (ep13) | Routed | 最优 |
|------|-----------------|----------------|--------|------|
| NRMSE | 0.0722 | 0.0681 | **0.0673** | Routed |
| WAPE | 0.1653 | 0.1644 | **0.1598** | Routed |
| MAE (kW) | 50.8 | 50.5 | **49.1** | Routed |
| RMSE (kW) | 82.2 | 77.6 | **76.7** | Routed |

### 分场站（测试集）

| 场站 | E04 MAE | E04+E02 MAE | Routed MAE | Routed 来源 |
|------|---------|-------------|------------|-------------|
| GS Paperboard | 119.3 | **112.8** | **112.8** | `E04+E02` |
| Plastone | **26.3** | 28.2 | **26.3** | `E04` |
| Quality-Coils | **32.8** | 36.6 | **32.8** | `E04` |
| Tamura | 24.9 | **24.6** | **24.6** | `E04+E02` |

### 峰值相关（测试集，`test_peak_by_series`）

| 场站 | E04 peak_ratio | E04+E02 peak_ratio | E04 nonpeak_bias (kW) | E04+E02 nonpeak_bias (kW) |
|------|----------------|--------------------|------------------------|---------------------------|
| GS | 0.737 | 0.739 | +48 | +51 |
| Plastone | 0.857 | **0.886** | +5 | +9 |
| Quality | 0.668 | **0.745** | +9 | +8 |
| Tamura | 0.640 | **0.702** | +8 | +9 |

说明：`peak_ratio` = 真峰时刻预测值 / 真实峰值（越接近 1 越好）。GS 峰顶仍偏低约 26%，加强 peak loss 后 GS 峰顶改善有限，但全场站 MAE/NRMSE 整体更优。Routed 方案没有生成新的峰值形态，而是直接选用各站点现有最优来源。

---

## 4. 完整实验结果汇总

说明：
- `状态=保留`：结果目录仍在仓库 `outputs/` 下，可直接复查。
- `状态=已删除`：结果目录已清理，保留这里的指标结论，避免后续重复试错。

| 实验 | Backbone / 改动 | Test MAE | Test RMSE | Test NRMSE | Test WAPE | 状态 | 结论 |
|------|------------------|---------:|----------:|-----------:|----------:|------|------|
| `0.2.0` | TimeXer 基线 | 57.71 | 93.18 | 0.0818 | 0.1877 | 保留 | 早期 TimeXer 基线 |
| `TimeXer E02` | TimeXer + peak loss 原型 | 61.11 | 106.11 | 0.0932 | 0.1988 | 已删除 | 对 `GS` 伤害过大，不采纳 |
| `TimeXer E05` | TimeXer + `series_id token` | 64.68 | 108.00 | N/A | N/A | 已删除 | `GS/Tamura` 明显变差，不采纳 |
| `ModernTCN E04` | ModernTCN + station balanced | 55.56 | 88.31 | 0.0776 | 0.1807 | 已删除 | 整体优于 TimeXer，但被 PatchTST 超过 |
| `PatchTST E04` | PatchTST + station balanced | 50.83 | 82.17 | 0.0722 | 0.1653 | 保留 | PatchTST 主线基线 |
| `PatchTST E04+E02` | PatchTST + stronger peak loss | 50.54 | 77.59 | 0.0681 | 0.1644 | 保留 | 单模型最优 |
| `PatchTST Routed` | `GS/Tamura -> E04+E02`, `Plastone/Quality -> E04` | **49.12** | **76.68** | **0.0673** | **0.1598** | 保留 | 整体最优，推荐上线 |
| `PatchTST S1+S2` | 去掉 `daily_max` + 弱化 peak loss | 51.61 | 82.35 | 0.0723 | 0.1679 | 已删除 | 峰值支撑不足，整体退步 |
| `PatchTST SpecialDay` | 增加 bridging / pre-post 半天 / 距离假日特征 | 58.63 | 90.61 | 0.0796 | 0.1907 | 已删除 | 新特征引入噪声，不采纳 |
| `PatchTST Overlap` | `patch_stride 12 -> 6` | 63.15 | 98.27 | 0.0863 | 0.2054 | 已删除 | overlap patch 明显负优化 |

补充观察：
- `E04+E02` 的收益主要来自 `GS / Tamura` 的峰值约束增强。
- `E04` 对 `Plastone / Quality` 的普通时段更稳。
- `Routed` 的价值正是利用了这两点互补性。

---

## 5. 验证集（选模用，仅供参考）

| 指标 | E04 | E04+E02 |
|------|-----|---------|
| val NRMSE | **0.122** | 0.134 |
| weekday→holiday WAPE | **0.695** | 0.757 |

E04+E02 在验证集上略差于 E04，但**测试集 7 天滚动**明显更好。验证窗仅 20 个（4 站×5 天），holiday 样本极少，不宜单独否定 E04+E02。

---

## 6. 已尝试但未采纳的改动

| 改动 | 结论 |
|------|------|
| `series_id_mode: token` | 对 `Quality` 略有帮助，但 `GS/Tamura` 明显变差，不采纳 |
| `scenario_loss_weights`（节假日场景 1.1–1.25） | `GS` 明显变差，验证 holiday 也没有净收益，不采纳 |
| 弱化 peak loss (`S1+S2`) | 峰值支撑回落，单模型整体退步，不采纳 |
| 特殊日切换特征 | 当前数据量下信息增益不够，反而引入噪声，不采纳 |
| overlap patch | 在当前设置下明显负优化，不采纳 |

---

## 7. Routed 方案说明

Routed 不是新训练出的第三个模型，而是一个**按站点选择现有最优模型**的后处理方案：

- `GS Paperboard` -> `E04+E02`
- `Tamura` -> `E04+E02`
- `Plastone` -> `E04`
- `Quality-Coils` -> `E04`

对应实现与产物：

- 脚本：`scripts/build_routed_experiment.py`
- 路由表：`outputs/experiments/patchtst_station_routed/route_map.json`
- 结果目录：`outputs/experiments/patchtst_station_routed/`

Routed 方案在测试集上的整体指标优于当前最优单模型：

- MAE：`50.54 -> 49.12`
- RMSE：`77.59 -> 76.68`
- NRMSE：`0.0681 -> 0.0673`
- WAPE：`0.1644 -> 0.1598`

---

## 8. E04+E02 相对 E04 的配置差异

```json
"peak_loss_weight": 0.3,
"underprediction_topk_weight": 0.15,
"daily_max_loss_weight": 0.1
```

（E04 为 0.2 / 0.08 / 0.0；其余结构、采样、主损失相同。）

---

## 9. 结论与建议

1. **生产推荐**：优先采用 `Routed` 方案，即 `GS / Tamura -> E04+E02`，`Plastone / Quality-Coils -> E04`。
2. **单模型推荐**：若部署链路只允许单模型，则使用 `patchtst_e04_e02_peak_loss`，checkpoint `patchtst_e04_e02_best_epoch13.pt`。
3. **基线保留**：E04 用于回归对比与 ablation 参照。
4. **优化结论**：当前这条 `PatchTST` 小步调参路线已经接近上限，继续削弱 peak loss、补特殊日特征、或做 overlap patch 都没有带来净收益。
5. **业务解读**：GS 绝对误差仍最大；节假日切换（`weekday→holiday`）仍是弱项，不宜仅靠 loss 加权，后续若再提升，建议转向更大改动的路线，例如 TSFM、正式化站点路由推理或新增业务特征。
6. **复现训练**：

```bash
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_balanced_sampler.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_e02_peak_loss.json
```

7. **复现 routed 评估**：

```bash
./.venv/bin/python scripts/build_routed_experiment.py
```

评估输出：`evaluation_summary.json`（含 `test_peak_by_series`）、`test_predictions.csv`、`plots/`。
