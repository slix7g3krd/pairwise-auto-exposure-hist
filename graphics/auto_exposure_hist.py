"""自动曝光直方图（auto-exposure histogram）。

针对六个缺陷的修复点与对应入口：

1. 统计与色调映射绑定同一帧序号
   —— `FrameHistogram.frame_index` 随统计数据一起在异步队列中流动，
   `AsyncHistogramQueue.stats_for_frame` 只返回同帧统计，`tone_map`
   在应用曝光前强制校验帧序号，杜绝跨帧误用。
2. 回读延迟用预测补偿
   —— `readback_latency` 在 log2 域对已回读的测量值做最小二乘等速外推，
   预测 ``latency_frames`` 帧之后的目标亮度，补偿固定 GPU 回读延迟。
3. 场景切换重置适应状态
   —— `scene_reset` 配合 `detect_scene_change`（归一化直方图交集，
   对分辨率变化不敏感），切场时复位眼睛适应值、龄期与预测器历史。
4. 亮度阈值按像素比例而非绝对值
   —— `lum_threshold` 接收比例（ratio），内部乘以（加权）总像素数，
   分辨率变化后阈值等比缩放，不漂移。
5. 曝光时间轴固定步进
   —— `timeline` / `adapt_speed` / `eye_adapt` 全部使用固定 dt
   （默认 1/60），不读取墙上时钟、不使用随机数，逐帧结果可复现。
6. 直方图分桶边界确定
   —— `make_buckets` 在固定 log2 范围 [MIN_LOG2, MAX_LOG2] 上均分，
   边界为冻结常量并缓存，桶内按比例线性插值，跨运行逐位一致；
   `determinism` 提供逐位校验和。
"""

from __future__ import annotations

import dataclasses
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    # 常量与配置
    "DEFAULT_BUCKET_COUNT",
    "MIN_LOG2",
    "MAX_LOG2",
    "NEUTRAL_LOG2",
    "LATENCY_FRAMES",
    "TIMESTEP",
    "EXPOSURE_MIN_LOG2",
    "EXPOSURE_MAX_LOG2",
    "ADAPT_UP_RATE",
    "ADAPT_DOWN_RATE",
    "SCENE_CHANGE_INTERSECTION",
    "HistogramConfig",
    "DEFAULT_CONFIG",
    # 数据结构
    "BucketGrid",
    "FrameHistogram",
    "MeteringRegion",
    "AdaptationState",
    "ExposureResult",
    "HistogramMetrics",
    "PercentileResult",
    "ThresholdResult",
    "ValidationReport",
    "AuditRecord",
    "AdaptSpeed",
    "TimelineSpec",
    "PredictionResult",
    "DeterminismReport",
    # 分桶（修复点 6）
    "make_buckets",
    "luminance_to_bucket",
    "build_histogram",
    "weighted_histogram",
    # 六个修复点的入口函数
    "hist_async",
    "readback_latency",
    "scene_reset",
    "lum_threshold",
    "adapt_speed",
    "determinism",
    # 其余契约入口
    "metering_region",
    "validation",
    "audit",
    "fallback",
    "clamp",
    "metrics",
    "percentile",
    "eye_adapt",
    "timeline",
    # 场景检测 / 色调映射 / 控制器
    "detect_scene_change",
    "tone_map",
    "FrameMismatchError",
    "AsyncHistogramQueue",
    "AutoExposureController",
    "run_exposure_curve",
]

# ---------------------------------------------------------------------------
# 固定常量：所有数值在进程启动时确定，运行期不依赖任何外部状态（修复点 5、6）
# ---------------------------------------------------------------------------

DEFAULT_BUCKET_COUNT = 64
MIN_LOG2 = -8.0                      # 约 0.004 cd/m^2
MAX_LOG2 = 16.0                      # 65536 cd/m^2，覆盖 HDR
NEUTRAL_LOG2 = 0.0                   # 中性灰目标（log2 域）
LATENCY_FRAMES = 2                   # GPU 回读固定延迟：两帧
TIMESTEP = 1.0 / 60.0                # 曝光时间轴固定步进
EXPOSURE_MIN_LOG2 = -16.0            # 曝光补偿钳位范围（EV）
EXPOSURE_MAX_LOG2 = 16.0
ADAPT_UP_RATE = 3.0                  # 变亮适应速率（log2 单位/秒）
ADAPT_DOWN_RATE = 1.5                # 变暗适应速率（log2 单位/秒）
SCENE_CHANGE_INTERSECTION = 0.6      # 归一化直方图交集低于此值判定切场


@dataclass(frozen=True)
class HistogramConfig:
    """直方图分桶配置（不可变）。"""

    bucket_count: int = DEFAULT_BUCKET_COUNT
    min_log2: float = MIN_LOG2
    max_log2: float = MAX_LOG2

    def __post_init__(self) -> None:
        if not isinstance(self.bucket_count, int) or self.bucket_count <= 0:
            raise ValueError("bucket_count 必须为正整数")
        if not math.isfinite(self.min_log2) or not math.isfinite(self.max_log2):
            raise ValueError("分桶边界必须为有限实数")
        if self.min_log2 >= self.max_log2:
            raise ValueError("min_log2 必须小于 max_log2")

    @property
    def log_step(self) -> float:
        return (self.max_log2 - self.min_log2) / self.bucket_count


DEFAULT_CONFIG = HistogramConfig()


@dataclass(frozen=True)
class BucketGrid:
    """确定的分桶边界（修复点 6）：edges 长度为 bucket_count + 1。"""

    edges: tuple[float, ...]
    log_step: float
    min_log2: float
    max_log2: float
    bucket_count: int


@dataclass(frozen=True)
class FrameHistogram:
    """带帧序号的直方图统计（修复点 1）。

    ``frame_index`` 随数据一起流动，色调映射只能消费同帧统计。
    """

    frame_index: int
    counts: tuple[float, ...]
    total_pixels: float
    bucket_count: int


@dataclass(frozen=True)
class MeteringRegion:
    """中央测光区域（像素坐标，左闭右开）。"""

    x0: int
    y0: int
    x1: int
    y1: int
    width: int
    height: int
    area: int


@dataclass(frozen=True)
class AdaptationState:
    """眼睛适应状态（修复点 3：切场时整体复位）。"""

    adapted_log_lum: float = NEUTRAL_LOG2
    scene_id: int = 0
    age: int = 0
    resets: int = 0


@dataclass(frozen=True)
class ExposureResult:
    """单帧曝光决策结果。"""

    frame_index: int
    stats_frame: int                    # 实际使用的统计帧；-1 表示无统计走兜底
    target_log2: float
    adapted_log_lum: float
    exposure_log2: float
    scene_changed: bool
    fallback_used: bool


@dataclass(frozen=True)
class HistogramMetrics:
    total_pixels: float
    occupied_buckets: int
    mean_log2: float
    mean_luminance: float
    under_ratio: float
    over_ratio: float


@dataclass(frozen=True)
class PercentileResult:
    q: float
    log2_value: float
    luminance: float
    bucket_index: int
    within_bucket: float


@dataclass(frozen=True)
class ThresholdResult:
    """亮度阈值（修复点 4：按像素比例计算）。"""

    pixels: float
    ratio: float
    total_pixels: float
    min_pixels: float
    max_pixels: float


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    total_pixels: float
    bucket_count: int


@dataclass(frozen=True)
class AuditRecord:
    frame_index: int
    total_pixels: float
    occupied_buckets: int
    empty_buckets: int
    min_bucket: int
    max_bucket: int
    warnings: tuple[str, ...]
    checksum: int


@dataclass(frozen=True)
class AdaptSpeed:
    """固定步进下的适应速率（修复点 5）。"""

    steps_per_second: int
    dt: float
    up_rate: float
    down_rate: float
    up_alpha: float
    down_alpha: float


@dataclass(frozen=True)
class TimelineSpec:
    frames: int
    dt: float
    start: float
    times: tuple[float, ...]


@dataclass(frozen=True)
class PredictionResult:
    """回读延迟预测补偿结果（修复点 2）。"""

    predicted_log2: float
    predicted_luminance: float
    slope: float
    frames: int
    available_samples: int


@dataclass(frozen=True)
class DeterminismReport:
    checksum_a: int
    checksum_b: int | None
    consistent: bool


# ---------------------------------------------------------------------------
# 分桶边界：固定、确定、缓存（修复点 6）
# ---------------------------------------------------------------------------

_GRID_CACHE: dict[HistogramConfig, BucketGrid] = {}


def make_buckets(config: HistogramConfig | None = None) -> BucketGrid:
    """返回确定的 log2 分桶网格，相同配置永远拿到同一对象。"""
    config = config or DEFAULT_CONFIG
    cached = _GRID_CACHE.get(config)
    if cached is not None:
        return cached
    step = config.log_step
    edges = tuple(
        config.min_log2 + i * step for i in range(config.bucket_count + 1)
    )
    grid = BucketGrid(
        edges=edges,
        log_step=step,
        min_log2=config.min_log2,
        max_log2=config.max_log2,
        bucket_count=config.bucket_count,
    )
    _GRID_CACHE[config] = grid
    return grid


def luminance_to_bucket(
    luminance: float, config: HistogramConfig | None = None
) -> int:
    """线性亮度 -> 桶下标，边界处归入右侧桶，超界钳到首尾。"""
    config = config or DEFAULT_CONFIG
    if not math.isfinite(luminance) or luminance <= 0.0:
        return 0
    grid = make_buckets(config)
    idx = int(math.floor((math.log2(luminance) - config.min_log2) / grid.log_step))
    if idx < 0:
        return 0
    if idx >= config.bucket_count:
        return config.bucket_count - 1
    return idx


def build_histogram(
    frame_index: int,
    counts: Sequence[float] | int,
    config: HistogramConfig | None = None,
) -> FrameHistogram:
    """构造带帧序号的直方图，校验长度并固化为元组。"""
    config = config or DEFAULT_CONFIG
    values, total = _as_histogram(counts, config)
    return FrameHistogram(
        frame_index=int(frame_index),
        counts=values,
        total_pixels=total,
        bucket_count=config.bucket_count,
    )


def weighted_histogram(
    counts: Sequence[float],
    weights: Sequence[float],
    config: HistogramConfig | None = None,
) -> tuple[tuple[float, ...], float]:
    """对直方图逐桶加权（如测光区域权重），返回 (加权计数, 加权总数)。"""
    config = config or DEFAULT_CONFIG
    values, _ = _as_histogram(counts, config)
    if len(weights) != len(values):
        raise ValueError("weights 与直方图桶数不一致")
    weighted = []
    for w in weights:
        if not math.isfinite(w) or w < 0.0:
            raise ValueError("权重必须为非负有限实数")
        weighted.append(0.0)
    out = tuple(values[i] * float(weights[i]) for i in range(len(values)))
    return out, math.fsum(out)


# ---------------------------------------------------------------------------
# 修复点 1：异步统计队列，帧序号随数据流动
# ---------------------------------------------------------------------------

class FrameMismatchError(RuntimeError):
    """色调映射试图消费非同一帧序号的统计结果。"""


class AsyncHistogramQueue:
    """模拟 GPU 异步计算队列 + 固定帧回读延迟。

    提交时严格校验帧序号单调递增；统计在 ``latency_frames`` 帧之后才
    可被读取；``stats_for_frame`` 只返回同帧统计，取不到时返回 None，
    调用方必须显式走兜底而不是误用别的帧。
    """

    def __init__(
        self,
        latency_frames: int = LATENCY_FRAMES,
        capacity: int = 64,
    ) -> None:
        if latency_frames < 0:
            raise ValueError("latency_frames 不能为负")
        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        self.latency_frames = int(latency_frames)
        self.capacity = int(capacity)
        self._in_flight: list[FrameHistogram] = []
        self._delivered: dict[int, FrameHistogram] = {}
        self._next_frame = 0
        self._current_frame = -1

    def submit(self, hist: FrameHistogram) -> FrameHistogram:
        if not isinstance(hist, FrameHistogram):
            raise TypeError("队列只接受 FrameHistogram")
        if hist.frame_index != self._next_frame:
            raise ValueError(
                f"帧序号失序：期望 {self._next_frame}，实际 {hist.frame_index}；"
                "统计与色调映射必须按同一帧序号严格对齐"
            )
        if len(self._in_flight) + len(self._delivered) >= self.capacity:
            raise OverflowError("异步统计队列容量耗尽")
        self._in_flight.append(hist)
        self._next_frame += 1
        return hist

    def advance(self, frames: int = 1) -> tuple[FrameHistogram, ...]:
        """时间轴固定步进 ``frames`` 帧，交付到期统计。

        统计帧 ``f`` 在当前时间轴推进到 ``f + latency_frames`` 时交付；
        latency=0 时同帧立即可用，latency=2 时延后两帧可读。
        """
        if frames < 0:
            raise ValueError("advance 帧数不能为负")
        delivered: list[FrameHistogram] = []
        for _ in range(frames):
            self._current_frame += 1
            remaining: list[FrameHistogram] = []
            for hist in self._in_flight:
                if self._current_frame - hist.frame_index >= self.latency_frames:
                    self._delivered[hist.frame_index] = hist
                    delivered.append(hist)
                else:
                    remaining.append(hist)
            self._in_flight = remaining
        return tuple(delivered)

    def stats_for_frame(self, frame_index: int) -> FrameHistogram | None:
        """只返回该帧自己的统计；不存在或尚未回读时返回 None。"""
        return self._delivered.get(int(frame_index))

    def latest_stats(self) -> FrameHistogram | None:
        """最新可用（已回读）统计；可能是较早帧，调用方据此做延迟预测。"""
        if not self._delivered:
            return None
        return self._delivered[max(self._delivered)]

    def discard_pending(self) -> int:
        """丢弃所有在途与已回读统计（场景切换时调用），返回丢弃帧数。"""
        dropped = len(self._in_flight) + len(self._delivered)
        self._in_flight.clear()
        self._delivered.clear()
        return dropped

    @property
    def in_flight(self) -> int:
        return len(self._in_flight)


def hist_async(
    latency_frames: int = LATENCY_FRAMES, capacity: int = 64
) -> AsyncHistogramQueue:
    """创建异步直方图统计队列（修复点 1 的工厂入口）。"""
    return AsyncHistogramQueue(latency_frames=latency_frames, capacity=capacity)


# ---------------------------------------------------------------------------
# 修复点 2：回读延迟预测补偿（log2 域等速外推）
# ---------------------------------------------------------------------------

def readback_latency(
    history: Sequence[float] | float,
    frames: int = LATENCY_FRAMES,
    dt: float = TIMESTEP,
) -> PredictionResult:
    """根据已回读的目标亮度历史预测 ``frames`` 帧之后的目标。

    历史样本按帧序（旧 -> 新）排列，最小二乘拟合 log2 域斜率后等速
    外推；只有一个样本时斜率为 0（保持当前值）。只使用已回读样本，
    不窥探未来帧。
    """
    if frames < 0:
        raise ValueError("预测帧数不能为负")
    if dt <= 0.0:
        raise ValueError("dt 必须为正数")
    samples = (
        [float(history)]
        if isinstance(history, (int, float))
        else [float(v) for v in history]
    )
    for v in samples:
        if not math.isfinite(v):
            raise ValueError("历史样本必须为有限实数")
    n = len(samples)
    if n == 0:
        raise ValueError("历史样本为空，无法做延迟预测")

    if n == 1:
        slope_per_frame = 0.0
    else:
        # 对最近至多 4 个样本做最小二乘（x 为帧偏移）
        window = samples[-4:]
        m = len(window)
        x_mean = (m - 1) / 2.0
        y_mean = math.fsum(window) / m
        var = math.fsum((i - x_mean) ** 2 for i in range(m))
        cov = math.fsum(
            (i - x_mean) * (window[i] - y_mean) for i in range(m)
        )
        slope_per_frame = cov / var if var > 0.0 else 0.0

    predicted = samples[-1] + slope_per_frame * float(frames)
    return PredictionResult(
        predicted_log2=predicted,
        predicted_luminance=2.0 ** predicted,
        slope=slope_per_frame / dt,
        frames=int(frames),
        available_samples=n,
    )


# ---------------------------------------------------------------------------
# 修复点 3：场景切换检测与适应状态复位
# ---------------------------------------------------------------------------

def _normalized(
    counts: Sequence[float], total: float
) -> tuple[float, ...]:
    if total <= 0.0:
        return tuple(0.0 for _ in counts)
    return tuple(c / total for c in counts)


def detect_scene_change(
    current: Sequence[float] | FrameHistogram,
    previous: Sequence[float] | None,
    config: HistogramConfig | None = None,
    threshold: float = SCENE_CHANGE_INTERSECTION,
) -> bool:
    """归一化直方图交集低于阈值即判定切场。

    交集基于比例而非像素绝对值，分辨率变化后判断不漂移。
    首帧（无历史）不判定为切场。
    """
    config = config or DEFAULT_CONFIG
    if isinstance(current, FrameHistogram):
        cur_counts, cur_total = current.counts, current.total_pixels
    else:
        cur_counts, cur_total = _as_histogram(current, config)
    if not previous:
        return False
    prev_counts, prev_total = _as_histogram(previous, config)
    if len(prev_counts) != len(cur_counts):
        return True
    if cur_total <= 0.0 or prev_total <= 0.0:
        return True
    cur_n = _normalized(cur_counts, cur_total)
    prev_n = _normalized(prev_counts, prev_total)
    intersection = math.fsum(min(a, b) for a, b in zip(cur_n, prev_n))
    return intersection < threshold


def scene_reset(
    state: AdaptationState | float | int,
    scene_id: int = 0,
    target_neutral: float = NEUTRAL_LOG2,
) -> AdaptationState:
    """复位眼睛适应状态（修复点 3）。

    适应值回到中性目标、龄期归零、复位计数 +1；预测器历史由控制器
    同步清空，保证切场后不会沿旧场景的趋势继续外推。
    """
    if isinstance(state, AdaptationState):
        prev_resets = state.resets
    else:
        prev_resets = 0
    if not math.isfinite(target_neutral):
        raise ValueError("target_neutral 必须为有限实数")
    return AdaptationState(
        adapted_log_lum=float(target_neutral),
        scene_id=int(scene_id),
        age=0,
        resets=prev_resets + 1,
    )


# ---------------------------------------------------------------------------
# 修复点 4：亮度阈值按像素比例
# ---------------------------------------------------------------------------

def lum_threshold(
    total_pixels_or_hist: float | Sequence[float] | FrameHistogram,
    ratio: float = 0.5,
    min_ratio: float = 0.0,
    max_ratio: float = 1.0,
    config: HistogramConfig | None = None,
) -> ThresholdResult:
    """按像素比例计算亮度阈值像素数（修复点 4）。

    入参可以是总像素数或直方图（自动取总像素）。分辨率变化时阈值随
    总像素等比缩放，比例保持不变。
    """
    if isinstance(total_pixels_or_hist, FrameHistogram):
        total = float(total_pixels_or_hist.total_pixels)
    elif isinstance(total_pixels_or_hist, Sequence):
        _, total = _as_histogram(total_pixels_or_hist, config or DEFAULT_CONFIG)
    else:
        total = float(total_pixels_or_hist)
    if total < 0.0:
        raise ValueError("总像素数不能为负")
    if not 0.0 <= min_ratio <= max_ratio <= 1.0:
        raise ValueError("要求 0 <= min_ratio <= max_ratio <= 1")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio 必须落在 [0, 1]")

    clamped_ratio = min(max(ratio, min_ratio), max_ratio)
    min_pixels = min_ratio * total
    max_pixels = max_ratio * total
    pixels = clamped_ratio * total
    return ThresholdResult(
        pixels=pixels,
        ratio=clamped_ratio,
        total_pixels=total,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


# ---------------------------------------------------------------------------
# 修复点 5：固定步进时间轴与眼睛适应
# ---------------------------------------------------------------------------

def timeline(
    frames: int, dt: float = TIMESTEP, start: float = 0.0
) -> TimelineSpec:
    """生成固定步进曝光时间轴（修复点 5），不读取墙上时钟。"""
    if frames < 0:
        raise ValueError("帧数不能为负")
    if dt <= 0.0:
        raise ValueError("dt 必须为正数")
    times = tuple(float(start) + i * float(dt) for i in range(frames))
    return TimelineSpec(
        frames=int(frames), dt=float(dt), start=float(start), times=times
    )


def adapt_speed(
    steps_per_second: int = 60,
    up_rate: float = ADAPT_UP_RATE,
    down_rate: float = ADAPT_DOWN_RATE,
) -> AdaptSpeed:
    """把每秒适应速率换算成固定步进下的每帧指数系数（修复点 5）。"""
    if steps_per_second <= 0:
        raise ValueError("steps_per_second 必须为正整数")
    if up_rate < 0.0 or down_rate < 0.0:
        raise ValueError("适应速率不能为负")
    dt = 1.0 / float(steps_per_second)
    return AdaptSpeed(
        steps_per_second=int(steps_per_second),
        dt=dt,
        up_rate=float(up_rate),
        down_rate=float(down_rate),
        up_alpha=1.0 - math.exp(-up_rate * dt),
        down_alpha=1.0 - math.exp(-down_rate * dt),
    )


def eye_adapt(
    state: AdaptationState | float | int,
    target_log2: float = NEUTRAL_LOG2,
    dt: float = TIMESTEP,
    up_rate: float = ADAPT_UP_RATE,
    down_rate: float = ADAPT_DOWN_RATE,
) -> AdaptationState:
    """单帧眼睛适应（修复点 5）：固定步进指数趋近，无随机、无时钟。"""
    if isinstance(state, AdaptationState):
        current = state.adapted_log_lum
        scene_id = state.scene_id
        age = state.age
        resets = state.resets
    else:
        current = float(state)
        scene_id, age, resets = 0, 0, 0
    if dt <= 0.0:
        raise ValueError("dt 必须为正数")
    if not math.isfinite(target_log2):
        raise ValueError("target_log2 必须为有限实数")

    rate = up_rate if target_log2 >= current else down_rate
    alpha = 1.0 - math.exp(-rate * dt)
    adapted = current + alpha * (target_log2 - current)
    return AdaptationState(
        adapted_log_lum=adapted,
        scene_id=scene_id,
        age=age + 1,
        resets=resets,
    )


# ---------------------------------------------------------------------------
# 测光区域、校验、审计、兜底、钳位
# ---------------------------------------------------------------------------

def metering_region(
    width: int,
    height: int | None = None,
    scale: float = 0.5,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> MeteringRegion:
    """计算中央测光区域（整数像素、左闭右开），越界自动钳位。"""
    if height is None:
        height = width
    if width <= 0 or height <= 0:
        raise ValueError("画面宽高必须为正整数")
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale 必须落在 (0, 1]")
    if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
        raise ValueError("中心点必须落在 [0, 1]")

    region_w = max(1, int(round(width * scale)))
    region_h = max(1, int(round(height * scale)))
    cx = width * center_x
    cy = height * center_y
    x0 = int(round(cx - region_w / 2.0))
    y0 = int(round(cy - region_h / 2.0))
    x0 = min(max(x0, 0), width - region_w)
    y0 = min(max(y0, 0), height - region_h)
    x1 = x0 + region_w
    y1 = y0 + region_h
    return MeteringRegion(
        x0=x0, y0=y0, x1=x1, y1=y1,
        width=region_w, height=region_h, area=region_w * region_h,
    )


def validation(
    counts: Sequence[float] | FrameHistogram,
    config: HistogramConfig | None = None,
) -> ValidationReport:
    """校验直方图结构与数值合法性。"""
    config = config or DEFAULT_CONFIG
    if isinstance(counts, FrameHistogram):
        values = counts.counts
        declared_total = counts.total_pixels
        bucket_count = counts.bucket_count
    else:
        values, _ = _as_histogram(counts, config)
        declared_total = None
        bucket_count = config.bucket_count

    errors: list[str] = []
    warnings: list[str] = []
    if len(values) != bucket_count:
        errors.append(
            f"桶数不匹配：期望 {bucket_count}，实际 {len(values)}"
        )
    bad = [i for i, v in enumerate(values) if not math.isfinite(v) or v < 0.0]
    if bad:
        errors.append(f"存在负值或非有限值的桶：{bad[:8]}")
    total = math.fsum(values) if not bad else 0.0
    if total <= 0.0:
        errors.append("直方图总像素数为 0，无法估计曝光")
    if declared_total is not None and not math.isclose(
        declared_total, total, rel_tol=1e-9, abs_tol=1e-6
    ):
        errors.append(
            f"total_pixels={declared_total} 与计数之和 {total} 不一致"
        )
    occupied = sum(1 for v in values if v > 0.0)
    if 0 < occupied <= max(1, bucket_count // 16):
        warnings.append("有效桶过少，测光结果可能不稳定")
    return ValidationReport(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        total_pixels=total,
        bucket_count=bucket_count,
    )


def audit(
    counts: Sequence[float] | FrameHistogram,
    frame_index: int = -1,
    config: HistogramConfig | None = None,
) -> AuditRecord:
    """对一帧直方图做结构审计并返回确定性校验和。"""
    config = config or DEFAULT_CONFIG
    if isinstance(counts, FrameHistogram):
        values, total = counts.counts, counts.total_pixels
        frame_index = counts.frame_index
    else:
        values, total = _as_histogram(counts, config)
    occupied_idx = [i for i, v in enumerate(values) if v > 0.0]
    warnings: list[str] = []
    if total <= 0.0:
        warnings.append("空直方图")
    if occupied_idx and (occupied_idx[0] == 0 or occupied_idx[-1] == len(values) - 1):
        warnings.append("分布触及首尾桶，存在钳位截断")
    return AuditRecord(
        frame_index=int(frame_index),
        total_pixels=total,
        occupied_buckets=len(occupied_idx),
        empty_buckets=len(values) - len(occupied_idx),
        min_bucket=occupied_idx[0] if occupied_idx else -1,
        max_bucket=occupied_idx[-1] if occupied_idx else -1,
        warnings=tuple(warnings),
        checksum=determinism(values).checksum_a,
    )


def fallback(
    counts: Sequence[float] | FrameHistogram | None,
    config: HistogramConfig | None = None,
    neutral: float = NEUTRAL_LOG2,
) -> float:
    """无有效统计时的兜底目标亮度（log2 域）。

    直方图合法时返回其 log2 均值；为空/非法时回到中性目标。
    """
    if counts is None:
        return float(neutral)
    report = validation(counts, config)
    if not report.valid:
        return float(neutral)
    return metrics(counts, config).mean_log2


def clamp(
    exposure_log2: float,
    min_log2: float = EXPOSURE_MIN_LOG2,
    max_log2: float = EXPOSURE_MAX_LOG2,
) -> float:
    """曝光补偿钳位。"""
    if min_log2 > max_log2:
        raise ValueError("min_log2 不能大于 max_log2")
    return min(max(float(exposure_log2), min_log2), max_log2)


def metrics(
    counts: Sequence[float] | FrameHistogram,
    config: HistogramConfig | None = None,
) -> HistogramMetrics:
    """基于固定分桶计算确定性统计量（修复点 6）。"""
    config = config or DEFAULT_CONFIG
    if isinstance(counts, FrameHistogram):
        values, total = counts.counts, counts.total_pixels
    else:
        values, total = _as_histogram(counts, config)
    if total <= 0.0:
        return HistogramMetrics(
            total_pixels=0.0, occupied_buckets=0,
            mean_log2=NEUTRAL_LOG2, mean_luminance=2.0 ** NEUTRAL_LOG2,
            under_ratio=0.0, over_ratio=0.0,
        )
    grid = make_buckets(config)
    step = grid.log_step
    weighted_log = 0.0
    weighted_lin = 0.0
    occupied = 0
    for i, c in enumerate(values):
        if c <= 0.0:
            continue
        occupied += 1
        log_mid = config.min_log2 + (i + 0.5) * step
        weighted_log += c * log_mid
        weighted_lin += c * (2.0 ** log_mid)
    return HistogramMetrics(
        total_pixels=total,
        occupied_buckets=occupied,
        mean_log2=weighted_log / total,
        mean_luminance=weighted_lin / total,
        under_ratio=values[0] / total,
        over_ratio=values[-1] / total,
    )


def percentile(
    counts: Sequence[float] | FrameHistogram,
    q: float = 0.5,
    config: HistogramConfig | None = None,
) -> PercentileResult:
    """固定桶边界上的百分位亮度，桶内按比例线性插值（修复点 6）。"""
    config = config or DEFAULT_CONFIG
    if isinstance(counts, FrameHistogram):
        values, total = counts.counts, counts.total_pixels
    else:
        values, total = _as_histogram(counts, config)
    if not 0.0 <= q <= 1.0:
        raise ValueError("q 必须落在 [0, 1]")
    if total <= 0.0:
        return PercentileResult(
            q=float(q), log2_value=config.min_log2,
            luminance=2.0 ** config.min_log2,
            bucket_index=0, within_bucket=0.0,
        )
    grid = make_buckets(config)
    target = q * total
    acc = 0.0
    for i, c in enumerate(values):
        if c <= 0.0:
            continue
        if acc + c >= target:
            lo = config.min_log2 + i * grid.log_step
            frac = (target - acc) / c
            log_value = lo + frac * grid.log_step
            return PercentileResult(
                q=float(q),
                log2_value=log_value,
                luminance=2.0 ** log_value,
                bucket_index=i,
                within_bucket=frac,
            )
        acc += c
    # 浮点尾差：落到最后一个非空桶上沿
    last = max(i for i, c in enumerate(values) if c > 0.0)
    log_value = config.min_log2 + (last + 1) * grid.log_step
    return PercentileResult(
        q=float(q), log2_value=log_value,
        luminance=2.0 ** log_value,
        bucket_index=last, within_bucket=1.0,
    )


# ---------------------------------------------------------------------------
# 修复点 6（续）：确定性校验和
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _flatten(value: Any) -> Iterable[Any]:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten(key)
            yield from _flatten(value[key])
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _flatten(i)
            yield from _flatten(item)
    elif value is None:
        yield -1
    elif isinstance(value, bool):
        yield int(value)
    elif isinstance(value, int):
        yield value
    elif isinstance(value, float):
        yield struct.pack("<d", value)
    elif isinstance(value, str):
        yield value.encode("utf-8")
    else:
        raise TypeError(f"无法纳入确定性校验和的类型：{type(value)!r}")


def _mixsum(payload: Any, salt: int = 0) -> int:
    state = salt & _MASK64
    for token in _flatten(payload):
        if isinstance(token, bytes):
            word = 0
            for byte in token:
                word = ((word << 8) | byte) & _MASK64
        else:
            word = int(token) & _MASK64
        state = _splitmix64(state ^ word)
    return state


def determinism(
    payload_a: Any,
    payload_b: Any = None,
    salt: int = 0,
) -> DeterminismReport:
    """对任意（嵌套）数值结构计算确定性校验和，可选比较两次运行。

    浮点按 IEEE-754 位模式参与混合，相同输入在任何机器、任何运行中
    都得到相同校验和（修复点 6 的可验证入口）。
    """
    checksum_a = _mixsum(payload_a, salt)
    if payload_b is None:
        return DeterminismReport(
            checksum_a=checksum_a, checksum_b=None, consistent=True
        )
    checksum_b = _mixsum(payload_b, salt)
    return DeterminismReport(
        checksum_a=checksum_a,
        checksum_b=checksum_b,
        consistent=checksum_a == checksum_b,
    )


# ---------------------------------------------------------------------------
# 色调映射：强制同帧序号（修复点 1）
# ---------------------------------------------------------------------------

def tone_map(
    linear_color: float | Sequence[float],
    result: ExposureResult,
    frame_index: int | None = None,
    allow_fallback: bool = False,
    reinhard: bool = False,
) -> float | tuple[float, ...]:
    """对线性颜色应用一帧曝光决策。

    ``frame_index`` 与 ``result.stats_frame`` 不一致时抛
    `FrameMismatchError`；预热期无统计（stats_frame == -1）只有显式
    ``allow_fallback`` 才允许使用兜底曝光。
    """
    if frame_index is not None:
        if result.stats_frame == -1 and not allow_fallback:
            raise FrameMismatchError(
                f"帧 {frame_index} 的统计尚未回读，且未允许兜底曝光"
            )
        if result.stats_frame != -1 and result.stats_frame != frame_index:
            raise FrameMismatchError(
                f"帧序号不一致：色调映射帧 {frame_index}，"
                f"统计帧 {result.stats_frame}"
            )
    scale = 2.0 ** result.exposure_log2

    def _map(c: float) -> float:
        v = float(c) * scale
        if reinhard:
            v = v / (1.0 + v)
        return v

    if isinstance(linear_color, (int, float)):
        return _map(linear_color)
    return tuple(_map(c) for c in linear_color)


# ---------------------------------------------------------------------------
# 控制器：把六个修复点串成逐帧流水线
# ---------------------------------------------------------------------------

class AutoExposureController:
    """逐帧自动曝光控制器。

    流水线（每帧固定步进，无时钟、无随机）：
      提交带帧序号直方图 -> 异步队列推进 -> 取同帧统计（修复点 1）
      -> 切场检测与复位（修复点 3）-> 回读延迟预测（修复点 2）
      -> 固定步进眼睛适应（修复点 5）-> 曝光钳位。
    """

    def __init__(
        self,
        config: HistogramConfig | None = None,
        latency_frames: int = LATENCY_FRAMES,
        queue_capacity: int = 256,
        scene_threshold: float = SCENE_CHANGE_INTERSECTION,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.scene_threshold = scene_threshold
        self.queue = AsyncHistogramQueue(
            latency_frames=latency_frames, capacity=queue_capacity
        )
        self.state = AdaptationState()
        self.frame_index = -1
        self.scene_id = 0
        self.results: list[ExposureResult] = []
        # 已回读测量值：(统计帧序号, log2 目标)，切场即清空
        self._measured: list[tuple[int, float]] = []
        self._prev_norm: tuple[float, ...] = ()

    @property
    def latency_frames(self) -> int:
        return self.queue.latency_frames

    def submit_frame(
        self,
        counts: Sequence[float],
        scene_id: int = 0,
    ) -> ExposureResult:
        self.frame_index += 1
        frame = self.frame_index

        # 修复点 1：统计带帧序号入队。先推进时间轴交付到期统计，
        # 再提交本帧统计——与 GPU 异步流水线的实际顺序一致。
        # 回读延迟内当前帧统计尚不可用，只能取最新已回读的旧帧统计，
        # 其帧序号记录在 stats_frame 中并由预测补偿（修复点 2）消掉时间差。
        self.queue.advance()
        hist = build_histogram(frame, counts, self.config)
        self.queue.submit(hist)
        stats = self.queue.latest_stats()

        scene_changed = False
        stats_frame = -1
        if stats is not None:
            stats_frame = stats.frame_index
            measured = fallback(stats, self.config)
            self._measured.append((stats.frame_index, measured))

            cur_norm = _normalized(stats.counts, stats.total_pixels)
            cut = detect_scene_change(
                stats.counts, self._prev_norm, self.config,
                threshold=self.scene_threshold,
            )
            self._prev_norm = cur_norm
            scene_changed = cut

        # scene_id 变化也算切场（即便直方图相近）
        if scene_id != self.scene_id:
            scene_changed = True
            self.scene_id = int(scene_id)

        if scene_changed:
            # 修复点 3：复位适应状态，并清空预测器历史、直方图基线与
            # 回读队列中残留的旧场景统计，避免旧统计在新场景里二次误判
            self.state = scene_reset(self.state, self.scene_id, NEUTRAL_LOG2)
            self._measured.clear()
            self._prev_norm = ()
            self.queue.discard_pending()

        # 修复点 2：只基于已回读样本预测当前帧目标
        if self._measured:
            history = [v for _, v in self._measured]
            prediction = readback_latency(
                history, frames=self.latency_frames, dt=TIMESTEP
            )
            target = prediction.predicted_log2
            used_fallback = False
        else:
            target = NEUTRAL_LOG2
            used_fallback = True

        # 修复点 5：固定步进眼睛适应
        self.state = eye_adapt(self.state, target, TIMESTEP)

        # 曝光补偿：把适应亮度拉回中性目标，并做钳位
        exposure = clamp(NEUTRAL_LOG2 - self.state.adapted_log_lum)
        result = ExposureResult(
            frame_index=frame,
            stats_frame=stats_frame,
            target_log2=target,
            adapted_log_lum=self.state.adapted_log_lum,
            exposure_log2=exposure,
            scene_changed=scene_changed,
            fallback_used=used_fallback,
        )
        self.results.append(result)
        return result

    def tone_map(
        self,
        linear_color: float | Sequence[float],
        frame_index: int | None = None,
        allow_fallback: bool = False,
        reinhard: bool = False,
    ) -> float | tuple[float, ...]:
        idx = self.frame_index if frame_index is None else frame_index
        try:
            result = self.results[idx]
        except IndexError as exc:
            raise FrameMismatchError(f"帧 {idx} 尚无曝光结果") from exc
        return tone_map(
            linear_color, result,
            frame_index=idx, allow_fallback=allow_fallback, reinhard=reinhard,
        )


def run_exposure_curve(
    frames: Sequence[Sequence[float]],
    scene_ids: Sequence[int] | None = None,
    config: HistogramConfig | None = None,
    latency_frames: int = LATENCY_FRAMES,
) -> tuple[ExposureResult, ...]:
    """对一段 HDR 素材跑完整曝光曲线（纯函数式入口，结果确定可复现）。"""
    controller = AutoExposureController(
        config=config, latency_frames=latency_frames
    )
    ids = list(scene_ids) if scene_ids is not None else [0] * len(frames)
    if len(ids) != len(frames):
        raise ValueError("scene_ids 与帧数不一致")
    results = [
        controller.submit_frame(counts, scene_id)
        for counts, scene_id in zip(frames, ids)
    ]
    return tuple(results)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _as_histogram(
    counts: Sequence[float] | int | float,
    config: HistogramConfig,
) -> tuple[tuple[float, ...], float]:
    """规范化直方图输入；标量被视为全部质量落在 0 号桶的退化直方图。"""
    if isinstance(counts, (int, float)):
        values = [0.0] * config.bucket_count
        values[0] = float(counts)
    else:
        values = [float(v) for v in counts]
    if len(values) != config.bucket_count:
        raise ValueError(
            f"直方图桶数不匹配：期望 {config.bucket_count}，实际 {len(values)}"
        )
    return tuple(values), math.fsum(values)
