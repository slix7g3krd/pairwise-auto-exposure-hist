# pairwise-auto-exposure-hist

## 领域

游戏、图形与多媒体

## 主题

自动曝光直方图

## 说明

异步统计、回读延迟与场景切换重置。

初始状态下 `graphics/auto_exposure_hist.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
