# Code Review: 负荷预测 TimeXer 实现

## 整体评价

代码结构清晰，模块拆分合理，与技术方案基本一致。但存在一个设计层面的问题影响了模型性能，以及若干细节需要关注。

---

## Issue #1: `ExogenousEmbedding` 丢失了外生特征的时序信息

[src/loadforecast/models/timexer.py#L106-L115](../src/loadforecast/models/timexer.py#L106-L115)

```python
class ExogenousEmbedding(nn.Module):
    def __init__(self, total_len: int, d_model: int, dropout: float):
        self.value_embedding = nn.Linear(total_len, d_model)  # total_len=2304

    def forward(self, history_exog, future_exog):
        x = torch.cat([history_exog, future_exog], dim=1)  # (B, 2304, 9)
        x = x.permute(0, 2, 1)                              # (B, 9, 2304)
        return self.dropout(self.value_embedding(x))         # (B, 9, 128)
```

`nn.Linear(2304, 128)` 将每个外生特征在整个时间窗口上的 2304 个时间步压缩为一个 128 维向量。这意味着 `minute_of_day_sin`、`minute_of_day_cos` 等编码一天内时刻信息的特征，其全部时序变化被折叠为单一向量。模型只能通过 patch 位置编码来间接推断时刻，但位置编码学到的是"第 N 个 patch"而非"几点几分"。

**这很可能是验证集 WAPE=0.53、NRMSE=0.46 且仅训练 1 个 epoch 就触发 early stopping 的根因**——模型无法有效利用日历协变量信息。

**建议修改方向**：
- 将 `ExogenousEmbedding` 改为保留时间维度，例如用 `nn.Linear(exog_dim, d_model)` 对每个时间步独立投影，输出 `(B, total_len, d_model)`
- 或者将外生特征按 patch 切分后与内生 patch 拼接，再一起送入 encoder

---

## Issue #2: 训练结果仅包含 1 个场站

`outputs/timexer_smoke/training_history.csv` 显示仅训练了 1 个 epoch，`best_epoch=1`。

`outputs/timexer_smoke/summary_metrics.json` 中 `val_by_series` 和 `test_by_series` 都只有 `Simpli-GS Paperboard` 一个场站，其余 3 个场站（Tamura、Plastone、Quality-Coils）未出现在输出中。

需要排查是 smoke test 只放了一个 CSV，还是 `build_daily_eval_specs` 把其他场站静默过滤掉了。[src/loadforecast/data.py#L204](../src/loadforecast/data.py#L204) 行 `if not all(candidate in full_day_set ...)` 会静默跳过不满足 7 天历史要求的日期，且没有任何 warning。

---

## Issue #3: `series_id` embedding 拼入外生特征的时机过早

[src/loadforecast/models/timexer.py#L248-L259](../src/loadforecast/models/timexer.py#L248-L259)

```python
def _append_series_embedding(self, history_exog, future_exog, series_id):
    series_embedding = self.series_embedding(series_id)          # (B, 2)
    history_static = series_embedding.unsqueeze(1).expand(-1, history_exog.shape[1], -1)
    future_static = series_embedding.unsqueeze(1).expand(-1, future_exog.shape[1], -1)
    return (
        torch.cat([history_exog, history_static], dim=-1),
        torch.cat([future_exog, future_static], dim=-1),
    )
```

`series_id` embedding 被拼接到外生特征后，再一起送入 `ExogenousEmbedding`，同样被 `nn.Linear(2304, 128)` 折叠了时序维度。`series_id` 是一个静态特征（不随时间变化），但当前的 Linear 层对每个时间步的 `series_id` 都学习独立的权重（2304 个参数仅仅为了编码一个常数），造成大量参数浪费。建议将 `series_id` 直接作为额外 token 加入到 encoder 中，而不是拼入外生特征。

---

## Issue #4: 未使用的 `factor` 配置项

[src/loadforecast/config.py#L36](../src/loadforecast/config.py#L36)

```python
factor: int = 5
```

`factor` 字段在 `ModelConfig` 中定义，但整个代码库中未引用。可能是从 Time-Series-Library 参考实现中遗留的字段，建议删除或补充实际用途。

---

## Issue #5: `resolve_station_holiday_subdiv` 的子串匹配可能误命中（已修复）

[src/loadforecast/data.py#L91-L92](../src/loadforecast/data.py#L91-L92)

```python
return cfg.station_holiday_subdiv_map.get(
    station_name, cfg.default_holiday_subdiv
)
```

此前该问题成立；当前实现已改为“完整站点名 -> 州代码”的精确映射，不再使用子串匹配，因此这条风险已消除。

---

## 其他观察

- **typo**: [src/loadforecast/models/timexer.py#L179](../src/loadforecast/models/timexer.py#L179) `self.endogenous_embedding` — 变量名拼写错误（不影响功能）
- **Huber delta=1.0** 作用于 RevIN 归一化后的空间（均值≈0，标准差≈1），取值合理
- 代码整体无安全漏洞、无数据泄露风险，训练/验证/测试的时间切分逻辑正确
- `requirements.txt` 建议固定版本号，避免后续环境不一致
