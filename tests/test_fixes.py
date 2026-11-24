"""六个修复点的专项行为测试。"""

import math

import pytest

import graphics.auto_exposure_hist as mod
from graphics.auto_exposure_hist import (
    DEFAULT_BUCKET_COUNT,
    LATENCY_FRAMES,
    NEUTRAL_LOG2,
    TIMESTEP,
    AdaptationState,
    AutoExposureController,
    AsyncHistogramQueue,
    FrameMismatchError,
    build_histogram,
    make_buckets,
)


def _hist(mid_bucket: int, spread: int = 2, pixels: int = 10000):
    """在 log2 桶 mid_bucket 附近生成三角分布直方图（确定性）。"""
    counts = [0] * DEFAULT_BUCKET_COUNT
    for offset in range(-spread, spread + 1):
        idx = mid_bucket + offset
        if 0 <= idx < DEFAULT_BUCKET_COUNT:
            counts[idx] = pixels * (spread + 1 - abs(offset))
    return counts


DARK = _hist(10)          # 暗场景
BRIGHT = _hist(50)        # 亮场景，与 DARK 的归一化直方图交集极低


# ---------------------------------------------------------------------------
# 修复点 1：统计与色调映射绑定同一帧序号
# ---------------------------------------------------------------------------

class TestFrameBinding:
    def test_queue_only_returns_same_frame_stats(self):
        queue = AsyncHistogramQueue(latency_frames=2)
        # 控制器语义：每帧先 advance 再 submit，统计在 f+latency 帧可读
        for i in range(4):
            queue.advance()
            queue.submit(build_histogram(i, _hist(20)))
        queue.advance()  # 推进到第 4 帧：满足 4-f>=2 的帧 0、1、2 交付
        for i in range(3):
            stats = queue.stats_for_frame(i)
            assert stats is not None and stats.frame_index == i
        assert queue.stats_for_frame(3) is None  # 帧 3 尚未回读
        # 最新可用统计是帧 2，且绝不会串成未来帧
        latest = queue.latest_stats()
        assert latest is not None and latest.frame_index == 2

    def test_submit_rejects_out_of_order_frame_index(self):
        queue = AsyncHistogramQueue(latency_frames=2)
        queue.submit(build_histogram(0, _hist(20)))
        with pytest.raises(ValueError, match="帧序号失序"):
            queue.submit(build_histogram(2, _hist(20)))

    def test_tone_map_rejects_mismatched_frame(self):
        ctrl = AutoExposureController()
        for i in range(4):
            ctrl.submit_frame(_hist(20))
        with pytest.raises(FrameMismatchError):
            ctrl.tone_map(1.0, frame_index=99)

    def test_warmup_requires_explicit_fallback_permission(self):
        ctrl = AutoExposureController(latency_frames=2)
        ctrl.submit_frame(_hist(20))  # 第 0 帧统计还没回读
        with pytest.raises(FrameMismatchError):
            ctrl.tone_map(1.0, frame_index=0)
        out = ctrl.tone_map(1.0, frame_index=0, allow_fallback=True)
        assert isinstance(out, float)


# ---------------------------------------------------------------------------
# 修复点 2：回读延迟预测补偿
# ---------------------------------------------------------------------------

class TestReadbackPrediction:
    def test_linear_ramp_extrapolates_forward(self):
        # 每帧 +1 的等速趋势，预测 2 帧后应外推到 5.0
        result = mod.readback_latency([0.0, 1.0, 2.0, 3.0], frames=2)
        assert result.predicted_log2 == pytest.approx(5.0)
        assert result.slope == pytest.approx(1.0 / TIMESTEP)
        assert result.predicted_luminance == pytest.approx(32.0)

    def test_single_sample_holds_value(self):
        result = mod.readback_latency([1.5], frames=LATENCY_FRAMES)
        assert result.slope == 0.0
        assert result.predicted_log2 == pytest.approx(1.5)

    def test_prediction_uses_only_past_samples(self):
        a = mod.readback_latency([0.0, 1.0, 2.0], frames=2)
        b = mod.readback_latency([0.0, 1.0, 2.0, 9.0], frames=2)
        # 加入新样本后预测值改变，说明预测严格基于已回读历史
        assert a.predicted_log2 != b.predicted_log2


# ---------------------------------------------------------------------------
# 修复点 3：场景切换重置适应状态
# ---------------------------------------------------------------------------

class TestSceneReset:
    def test_reset_restores_neutral_and_increments_count(self):
        state = AdaptationState(adapted_log_lum=4.2, age=30, resets=1)
        reset = mod.scene_reset(state, scene_id=7)
        assert reset.adapted_log_lum == NEUTRAL_LOG2
        assert reset.age == 0
        assert reset.scene_id == 7
        assert reset.resets == 2

    def test_detect_scene_change_uses_normalized_intersection(self):
        assert mod.detect_scene_change(BRIGHT, DARK) is True
        assert mod.detect_scene_change(DARK, DARK) is False
        # 首帧无历史不判切场
        assert mod.detect_scene_change(DARK, None) is False

    def test_controller_marks_scene_change_and_resets(self):
        # 零延迟：统计同帧提交、下一帧可读，测试聚焦复位逻辑本身
        ctrl = AutoExposureController(latency_frames=0)
        for _ in range(7):
            ctrl.submit_frame(DARK)
        before = ctrl.state.adapted_log_lum
        assert before != NEUTRAL_LOG2  # 暗场已让适应值偏离中性
        # DARK->BRIGHT 归一化直方图交集极低，且 scene_id 切换
        result = ctrl.submit_frame(BRIGHT, scene_id=1)
        assert result.scene_changed is True
        assert ctrl.state.age == 1
        assert ctrl.state.resets == 1
        # 复位后第一步的适应值从中性重新开始，而非沿旧值继续
        assert ctrl.state.adapted_log_lum != before

    def test_reset_clears_predictor_history(self):
        ctrl = AutoExposureController(latency_frames=0)
        for _ in range(5):
            ctrl.submit_frame(DARK)
        assert len(ctrl._measured) > 0
        ctrl.submit_frame(BRIGHT, scene_id=1)
        assert ctrl._measured == []  # 预测器历史被清空，不沿旧趋势外推


# ---------------------------------------------------------------------------
# 修复点 4：亮度阈值按像素比例而非绝对值
# ---------------------------------------------------------------------------

class TestRatioThreshold:
    def test_threshold_scales_with_resolution(self):
        lo = mod.lum_threshold(1920 * 1080, ratio=0.5)
        hi = mod.lum_threshold(3840 * 2160, ratio=0.5)
        assert lo.ratio == hi.ratio == 0.5
        assert hi.pixels == pytest.approx(4.0 * lo.pixels)

    def test_threshold_accepts_histogram_input(self):
        result = mod.lum_threshold(DARK, ratio=0.25)
        assert result.total_pixels == pytest.approx(sum(DARK))
        assert result.pixels == pytest.approx(0.25 * sum(DARK))

    def test_ratio_clamped_to_bounds(self):
        result = mod.lum_threshold(1000.0, ratio=0.9, min_ratio=0.1, max_ratio=0.5)
        assert result.ratio == 0.5
        assert result.pixels == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 修复点 5：曝光时间轴固定步进
# ---------------------------------------------------------------------------

class TestFixedTimeline:
    def test_timeline_uses_fixed_step(self):
        spec = mod.timeline(5)
        assert spec.dt == TIMESTEP
        assert spec.times == pytest.approx([i * TIMESTEP for i in range(5)])

    def test_adapt_speed_per_frame_alpha(self):
        speed = mod.adapt_speed(60, up_rate=3.0, down_rate=1.5)
        assert speed.up_alpha == pytest.approx(1.0 - math.exp(-3.0 / 60.0))
        assert speed.down_alpha == pytest.approx(1.0 - math.exp(-1.5 / 60.0))

    def test_eye_adapt_deterministic_fixed_step(self):
        s1 = mod.eye_adapt(0.0, 4.0)
        s2 = mod.eye_adapt(0.0, 4.0)
        assert s1.adapted_log_lum == s2.adapted_log_lum
        assert s1.age == 1
        # 固定步进下两步连走与一次 dt=2*TIMESTEP 的连续指数结果一致
        two_steps = mod.eye_adapt(s1, 4.0).adapted_log_lum
        one_double = mod.eye_adapt(0.0, 4.0, dt=2.0 * TIMESTEP).adapted_log_lum
        assert two_steps == pytest.approx(one_double)


# ---------------------------------------------------------------------------
# 修复点 6：直方图分桶边界确定 + 整体可复现
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_bucket_edges_are_fixed_and_cached(self):
        grid_a = make_buckets()
        grid_b = make_buckets()
        assert grid_a is grid_b
        assert len(grid_a.edges) == DEFAULT_BUCKET_COUNT + 1
        assert grid_a.edges[0] == mod.MIN_LOG2
        assert grid_a.edges[-1] == mod.MAX_LOG2
        # 等距分桶
        diffs = {
            round(grid_a.edges[i + 1] - grid_a.edges[i], 12)
            for i in range(DEFAULT_BUCKET_COUNT)
        }
        assert len(diffs) == 1

    def test_checksum_stable_across_calls(self):
        payload = {"frame": 3, "counts": DARK, "ev": -1.25}
        r1 = mod.determinism(payload)
        r2 = mod.determinism(payload)
        assert r1.checksum_a == r2.checksum_a
        diff = mod.determinism(payload, {**payload, "ev": -1.5})
        assert diff.consistent is False

    def test_same_hdr_material_produces_identical_curve(self):
        material = [
            _hist(b) for b in (10, 12, 15, 20, 26, 34, 44, 50, 48, 40)
        ]
        run_a = mod.run_exposure_curve(material)
        run_b = mod.run_exposure_curve(material)
        assert len(run_a) == len(run_b) == 10
        for a, b in zip(run_a, run_b):
            assert a == b
        # 校验和同样一致
        assert mod.determinism(run_a, run_b).consistent is True

    def test_different_material_produces_different_curve(self):
        # 零延迟：统计下一帧即生效，素材差异必须体现在曝光曲线上
        run_a = mod.run_exposure_curve(
            [_hist(b) for b in (10, 12, 15, 18)], latency_frames=0
        )
        run_b = mod.run_exposure_curve(
            [_hist(b) for b in (40, 42, 45, 48)], latency_frames=0
        )
        assert mod.determinism(run_a, run_b).consistent is False
