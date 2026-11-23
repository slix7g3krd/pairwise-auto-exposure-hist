import pytest
import graphics.auto_exposure_hist as mod

CASES = [
    ("hist_async", (1,)),
    ("readback_latency", (1,)),
    ("scene_reset", (1,)),
    ("lum_threshold", (1,)),
    ("adapt_speed", (1,)),
    ("determinism", (1,)),
    ("metering_region", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("clamp", (1,)),
    ("metrics", (1,)),
    ("percentile", (1,)),
    ("eye_adapt", (1,)),
    ("timeline", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
