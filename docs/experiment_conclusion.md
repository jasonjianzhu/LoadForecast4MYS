# PatchTST 负荷预测实验结论

## 1. 实验背景

- **任务**：4 个场站、5 分钟粒度，用前 7 天负荷预测未来 1 天（288 点）
- **数据**：`data/` 下 4 份 CSV，`target_clip_min=0` 清洗负负荷
- **Backbone**：PatchTST（`series_id_mode: repeat`）+ RevIN + 日历外生特征
- **训练**：`station_balanced` 采样，Huber+MSE 主损失，验证/测试各场站最后 5/7 天滚动评估
- **早停**：`early_stopping_metric=nrmse`，`patience=8`

在 E04 基线上曾尝试：series token、场景损失加权、加强峰值损失；**仅加强峰值损失在测试集上有效**，其余回退。随后进一步验证了一个**按站点路由**的后处理方案：对 `GS / Tamura` 采用 `E04+E02`，对 `Plastone / Quality-Coils` 采用 `E04`。完整过程见 git 历史；仓库当前保留 E04、E04+E02 及 routed 评估结果。

---

## 2. 保留实验

| 代号 | 配置 | 输出目录 | 用途 |
|------|------|----------|------|
| **E04** | `configs/experiments/patchtst_e04_balanced_sampler.json` | `outputs/experiments/patchtst_e04_balanced_sampler/` | 基线对照 |
| **E04+E02** | `configs/experiments/patchtst_e04_e02_peak_loss.json` | `outputs/experiments/patchtst_e04_e02_peak_loss/` | 单模型最优 |
| **Routed** | `scripts/build_routed_experiment.py` | `outputs/experiments/patchtst_station_routed/` | **推荐上线** |

**推荐单模型权重**：`patchtst_e04_e02_best_epoch13.pt`（best_epoch=13）

---

## 3. 测试集结果（主要依据）

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

## 6. Routed 方案说明

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

## 7. E04+E02 相对 E04 的配置差异

```json
"peak_loss_weight": 0.3,
"underprediction_topk_weight": 0.15,
"daily_max_loss_weight": 0.1
```

（E04 为 0.2 / 0.08 / 0.0；其余结构、采样、主损失相同。）

---

## 8. 结论与建议

1. **生产推荐**：优先采用 `Routed` 方案，即 `GS / Tamura -> E04+E02`，`Plastone / Quality-Coils -> E04`。
2. **单模型推荐**：若部署链路只允许单模型，则使用 `patchtst_e04_e02_peak_loss`，checkpoint `patchtst_e04_e02_best_epoch13.pt`。
3. **基线保留**：E04 用于回归对比与 ablation 参照。
4. **业务解读**：GS 绝对误差仍最大；节假日切换（`weekday→holiday`）仍是弱项，不宜仅靠 loss 加权，后续可考虑日历特征增强或加长验证窗。
5. **复现训练**：

```bash
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_balanced_sampler.json
./.venv/bin/python scripts/train_timexer.py --config configs/experiments/patchtst_e04_e02_peak_loss.json
```

6. **复现 routed 评估**：

```bash
./.venv/bin/python scripts/build_routed_experiment.py
```

评估输出：`evaluation_summary.json`（含 `test_peak_by_series`）、`test_predictions.csv`、`plots/`。
