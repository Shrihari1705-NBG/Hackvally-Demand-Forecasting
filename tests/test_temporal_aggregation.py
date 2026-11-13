import numpy as np
import pandas as pd

def test_asia_kolkata_180_day_period():
    end = pd.Timestamp("2026-11-12", tz="Asia/Kolkata").normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=180)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    assert len(hours) == 180 * 24
    assert str(hours.tz) == "Asia/Kolkata"

def test_deterministic_random_generation():
    assert np.array_equal(np.random.default_rng(42).poisson(3, 20), np.random.default_rng(42).poisson(3, 20))

def test_zero_demand_hour_is_retained():
    hours = pd.date_range("2026-01-01", periods=2, freq="h", tz="Asia/Kolkata")
    full = pd.MultiIndex.from_product([["cell-a"], hours], names=["pickup_h3", "hour_start_local"]).to_frame(index=False)
    observed = pd.DataFrame({"pickup_h3": ["cell-a"], "hour_start_local": [hours[0]], "request_count": [2]})
    result = full.merge(observed, on=["pickup_h3", "hour_start_local"], how="left")
    result["request_count"] = result["request_count"].fillna(0).astype(int)
    assert result.request_count.tolist() == [2, 0]
