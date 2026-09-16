import numpy as np
import pytest

from annealctrl.schedules import Schedule, decode_durations, inverse_density_schedule, pause_schedule, slow_window_schedule, window_schedule


def test_linear_and_query_validation():
    schedule = Schedule.linear()
    np.testing.assert_allclose(schedule([0, 0.2, 1]), [0, 0.2, 1])
    assert schedule(0.5) == 0.5
    with pytest.raises(ValueError):
        schedule(-0.01)
    with pytest.raises(ValueError):
        schedule(np.nan)
    with pytest.raises(ValueError):
        schedule.tau_knots[0] = 2


@pytest.mark.parametrize("tau,s", [([0, 0.5, 0.5, 1], [0, 0.3, 0.4, 1]), ([0, 0.5, 1], [0, 1.1, 1]), ([0, 1], [0.1, 1]), ([0, 1], [0, np.nan])])
def test_invalid_schedule(tau, s):
    with pytest.raises(ValueError):
        Schedule(tau, s)


def test_duration_decoder_feasible_and_units():
    rng = np.random.default_rng(4)
    for _ in range(30):
        schedule = decode_durations(rng.normal(size=8), runtime=20, max_slope=0.2)
        assert schedule.tau_knots[-1] == 1
        assert np.all(schedule.slopes(20) <= 0.2 * (1 + 1e-12))
    # Same dimensionless runtime*slope gives the same normalized waveform.
    a = decode_durations([1, -1, 0], runtime=20, max_slope=0.2)
    b = decode_durations([1, -1, 0], runtime=2, max_slope=2)
    np.testing.assert_allclose(a.tau_knots, b.tau_knots)


def test_runtime_boundary_overflow_and_gauge():
    schedule = decode_durations([10000, -10000], runtime=2, max_slope=1)
    assert np.isfinite(schedule.tau_knots).all()
    schedule.validate_slope(2, 1)
    a = decode_durations([1, -2, 0])
    b = decode_durations([1001, 998, 1000])
    np.testing.assert_allclose(a.tau_knots, b.tau_knots)
    fastest = decode_durations([10, -10], runtime=0.5, max_slope=2)
    np.testing.assert_allclose(fastest.tau_knots, [0, 0.5, 1])
    with pytest.raises(ValueError, match="infeasible"):
        decode_durations([0, 0], runtime=0.49, max_slope=2)


def test_repeated_path_knots_are_real_pauses():
    schedule = decode_durations([0, 0, 0], [0, 0.4, 0.4, 1], runtime=2, max_slope=1)
    a, b = schedule.tau_knots[1:3]
    assert b > a
    np.testing.assert_allclose(schedule(np.linspace(a, b, 7)), 0.4)
    no_spare = decode_durations([0, 0, 0], [0, 0.4, 0.4, 1], runtime=1, max_slope=1)
    assert len(no_spare.tau_knots) == 3
    np.testing.assert_allclose(no_spare(np.linspace(0, 1, 10)), np.linspace(0, 1, 10))


def test_windows_and_pauses():
    window = slow_window_schedule(0.3, 0.6, 0.8, runtime=2, max_slope=1)
    window.validate_slope(2, 1)
    # The middle segment receives the extra time and is slower.
    assert window.slopes(2)[1] < window.slopes(2)[0]
    double = window_schedule([(0.2, 0.3), (0.7, 0.8)], [0.4, 0.4], runtime=2, max_slope=1)
    double.validate_slope(2, 1)
    pause = pause_schedule(0.4, 0.25, runtime=2, max_slope=1)
    assert np.diff(pause.tau_knots)[1] == pytest.approx(0.25)
    assert pause.slopes(2)[1] == 0
    with pytest.raises(ValueError):
        pause_schedule(0.4, 0.6, runtime=2, max_slope=1)
    with pytest.raises(ValueError):
        slow_window_schedule(0.3, 0.3, 0.5)


def test_inverse_density():
    grid = np.linspace(0, 1, 101)
    schedule = inverse_density_schedule(grid, np.ones_like(grid))
    np.testing.assert_allclose(schedule(grid), grid, atol=1e-14)
    scaled = inverse_density_schedule(grid, np.ones_like(grid) * 1e300)
    np.testing.assert_allclose(scaled.tau_knots, schedule.tau_knots)
    with pytest.raises(ValueError):
        inverse_density_schedule([0, 0.5, 1], [1, 0, 1])
