# PoreMind

API-first 单分子纳米孔分析工具：

- 输入 ABF/CSV 电流轨迹
- 预处理 + 基线估计 + 事件检测
- 输出逐事件 DataFrame（时间范围、统计特征）
- 支持用户标注标准品训练分类器
- 支持训练后模型对新样本逐事件判定

## Quick start

```python
from poremind.pipeline import AnalysisConfig, analyze_abf_to_event_df

cfg = AnalysisConfig(reader="abf")
df = analyze_abf_to_event_df("example.abf", config=cfg, channel=0, sweep=0)
```

详细逐步流程见：`notebooks/step_by_step_analysis.ipynb`
