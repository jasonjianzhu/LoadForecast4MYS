# 负荷预测方案（0.4.1）

## 1. 任务定义

- 数据目录：`data/`
- 场站数量：4 个
- 目标字段：`load(kW)`
- 时间粒度：5 分钟
- 输入窗口：前 7 天，共 `2016` 点
- 预测窗口：未来 1 天，共 `288` 点

当前任务是标准的 day-ahead 负荷预测：

- 预测第 `t` 天时，输入为前 `7` 个完整自然日的真实负荷
- 输出为第 `t` 天 `00:00-23:55` 的完整 288 点曲线
- 验证和测试都按天滚动，不跨天

说明：

- 测试阶段采用滚动窗口评估，而不是递推 7 天预测
- 例如预测第 9 天时，输入是第 2-8 天的真实值，不会把第 8 天预测值回灌

## 2. 数据处理

对每个 CSV 统一执行以下处理：

1. 读取 `Time` 和 `load(kW)`
2. 时间戳标准化并按时间排序
3. 去重，重复时刻取均值
4. 对齐到完整的 5 分钟时间轴
5. 只保留完整自然日进入验证/测试
6. 目标值做下界裁剪：`target_clip_min = 0.0`

当前站点假日归属：

- `Simpli-Quality-Coils` -> `Perak (PRK)`
- 其他站点 -> `Selangor (SGR)`

## 3. 样本构造与切分

### 3.1 训练样本

- 输入：`load[t-2016 : t)`
- 输出：`load[t : t+288)`
- 训练步长：`train_stride = 6`

即训练阶段每 30 分钟切一个监督样本。

### 3.2 验证与测试

- 每个场站最后 `7` 个完整自然日作为测试集
- 测试集之前的 `5` 个完整自然日作为验证集
- 每个验证/测试样本都严格按天对齐

切分逻辑在：

- `src/loadforecast/data.py`

## 4. 外生特征

当前版本只保留验证过有效的基础日历特征，共 9 维：

- `minute_of_day_sin`
- `minute_of_day_cos`
- `day_of_week_sin`
- `day_of_week_cos`
- `is_weekend`
- `is_federal_holiday`
- `is_state_holiday`
- `is_pre_holiday`
- `is_post_holiday`

说明：

- 已尝试更细的“特殊日切换”特征，如 bridging holiday、节前下午、节后上午、距假日天数等，但在当前数据规模下带来净退步，因此 0.4.1 不采用
- 不使用未来不可知的 `PV(kW)`、`Meter(kW)` 等变量

## 5. 模型结构

0.4.1 的主线模型为 `PatchTST + RevIN`。

### 5.1 PatchTST

当前统一结构配置：

- `backbone = patchtst`
- `d_model = 192`
- `d_ff = 384`
- `e_layers = 4`
- `n_heads = 4`
- `dropout = 0.1`
- `patch_len = 12`
- `patch_stride = 12`
- `series_id_embedding_dim = 2`
- `series_id_mode = repeat`

说明：

- 已尝试 overlap patch（`patch_stride = 6`），测试集明显退步，因此 0.4.1 不采用
- 已尝试 `series_id token`，对 `Quality` 略有帮助，但 `GS/Tamura` 明显变差，因此不采用

### 5.2 RevIN

当前统一配置：

- `revin_affine = false`
- `revin_per_station_affine = false`
- `revin_eps = 1e-5`

RevIN 只作用在目标序列 `load(kW)` 上，日历协变量不参与 RevIN。

## 6. 训练方式

### 6.1 采样

训练使用站点均衡采样：

- `train_sampler = station_balanced`
- `station_balance_power = 0.5`

作用：

- 轻度提升样本较少站点的曝光次数
- 不直接通过 loss 对站点做强加权

### 6.2 优化器与训练超参数

统一配置：

- `optimizer = AdamW`
- `learning_rate = 1e-3`
- `weight_decay = 1e-4`
- `batch_size = 16`
- `eval_batch_size = 32`
- `max_epochs = 60`
- `early_stopping_patience = 8`
- `early_stopping_metric = nrmse`
- `scheduler = ReduceLROnPlateau`
- `grad_clip_norm = 1.0`

## 7. 损失函数

0.4.1 有两套实际保留的单模型配置：`E04` 和 `E04+E02`。

### 7.1 公共主损失

两者主损失相同，都是：

```text
L_pointwise = 0.7 * Huber + 0.3 * MSE
```

补充配置：

- `base_loss = hybrid`
- `huber_delta = 1.0`
- `pointwise_peak_focus_weight = 0.0`
- `underprediction_weight = 0.0`

### 7.2 E04 基线

`E04` 在主损失外，再加较温和的峰值约束：

```text
L_total
= L_pointwise
+ 0.2  * L_peak_topk
+ 0.08 * L_underprediction_topk
```

其中：

- `peak_top_k = 24`
- `daily_max_loss_weight = 0.0`

特点：

- 对 `Plastone / Quality` 的普通时段更稳
- 作为当前主线基线保留

### 7.3 E04+E02 单模型最优

`E04+E02` 在 `E04` 基础上加强峰值约束：

```text
L_total
= L_pointwise
+ 0.3  * L_peak_topk
+ 0.15 * L_underprediction_topk
+ 0.1  * L_daily_max
```

其中：

- `peak_top_k = 24`
- `daily_max_loss_weight = 0.1`

特点：

- 对 `GS / Tamura` 的峰值支撑更强
- 当前单模型最优

说明：

- 已尝试进一步弱化 peak loss（`S1+S2`），整体退步
- 已尝试 overlap patch、特殊日扩展特征，也都退步

## 8. 当前推荐方法

### 8.1 生产推荐：Routed

0.4.1 的整体最优方案不是单个 checkpoint，而是按站点路由的后处理方案：

- `GS / Tamura` -> 使用 `E04+E02`
- `Plastone / Quality-Coils` -> 使用 `E04`

路由脚本：

- `scripts/build_routed_experiment.py`

该方案的核心思想是：

- `E04+E02` 更擅长 `GS / Tamura` 的峰值约束
- `E04` 更适合 `Plastone / Quality` 的普通时段稳定性

### 8.2 单模型推荐

如果部署链路只允许一个模型，则使用：

- `PatchTST E04+E02`
- checkpoint：`patchtst_e04_e02_best_epoch13.pt`

## 9. 评估指标

统一输出以下指标：

- `MAE`
- `RMSE`
- `NMAE`
- `NRMSE`
- `WAPE`
- `MAPE`

定义口径：

- `NMAE = MAE / (y_max - y_min)`
- `NRMSE = RMSE / (y_max - y_min)`
- `WAPE = sum(|error|) / sum(|y_true|)`
- `MAPE = mean(|error| / max(|y_true|, 1.0))`

此外还单独监控峰值诊断：

- `peak_ratio_mean`
- `peak_err_mean`
- `nonpeak_bias_mean`

## 10. 0.4.1 结果概览

### 10.1 单模型

- `E04`
  - `MAE = 50.83`
  - `RMSE = 82.17`
  - `NRMSE = 0.0722`
  - `WAPE = 0.1653`

- `E04+E02`
  - `MAE = 50.54`
  - `RMSE = 77.59`
  - `NRMSE = 0.0681`
  - `WAPE = 0.1644`

### 10.2 Routed

- `MAE = 49.12`
- `RMSE = 76.68`
- `NRMSE = 0.0673`
- `WAPE = 0.1598`

结论：

- `E04+E02` 是单模型最优
- `Routed` 是整体最优，也是 0.4.1 的生产推荐

## 11. 图表资产

0.4.1 routed 方案的四张单站测试图已转存到可跟踪目录：

- `docs/figures/0.4.1/simpli_gs_paperboard_and_packaging_sdn_bhd_test_curve.svg`
- `docs/figures/0.4.1/simpli_plastone_technolngy_packaging_sdn_bhd_test_curve.svg`
- `docs/figures/0.4.1/simpli_quality_coils_test_curve.svg`
- `docs/figures/0.4.1/tamura_electronics_test_curve.svg`

## 12. 后续建议

在当前数据、特征和监督框架下，小步调参已经接近上限。继续提升时，优先级建议为：

1. 正式化 routed 推理链路
2. 评估 TSFM 路线，如 `Chronos-2` / `TimesFM`
3. 引入更强业务特征，如班次、停机计划、工艺日历

不建议继续沿当前 `PatchTST` 主线做以下方向：

- 继续弱化 peak loss
- 特殊日切换特征扩展
- overlap patch
- `series_id token`
