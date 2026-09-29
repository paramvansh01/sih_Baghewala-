import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from twin.physics.thermal import marx_langenheim_area
from twin.physics.wellbore import production_temperature_profile, steam_heat_loss_rate

Q, H, M, LAM, ALPHA, DT = 3.0e6, 15.0, 2.4e6, 1.7, 7e-7, 250.0


def _numerical_ml(t_end, n=4000):
    """Independent solution of Q t = M h dT A(t) + sum_j dA_j * 2 * 2 lam dT sqrt((t-tau_j)/(pi alpha))
    (cumulative heat balance; 2 faces), solved step by step for the area increments."""
    ts = np.linspace(0, t_end, n + 1)
    taus, dAs = [], []
    for k in range(1, n + 1):
        t = ts[k]
        tau_k = 0.5 * (ts[k - 1] + ts[k])
        loss_prev = sum(dA * 4 * LAM * DT * np.sqrt((t - tau) / (np.pi * ALPHA)) for tau, dA in zip(taus, dAs))
        coef = M * H * DT + 4 * LAM * DT * np.sqrt((t - tau_k) / (np.pi * ALPHA))
        a_prev = sum(dAs)
        dA = (Q * t - M * H * DT * a_prev - loss_prev) / coef
        taus.append(tau_k)
        dAs.append(dA)
    return sum(dAs)


@pytest.mark.parametrize("days", [2, 10, 40])
def test_marx_langenheim_matches_numerical_heat_balance(days):
    t = days * 86400.0
    a_num = _numerical_ml(t, n=800)
    a_ml = marx_langenheim_area(Q, t, H, M, LAM, ALPHA, DT)
    assert a_ml == pytest.approx(a_num, rel=0.01)


def test_marx_langenheim_short_time_limit_no_losses():
    t = 60.0
    assert marx_langenheim_area(Q, t, H, M, LAM, ALPHA, DT) == pytest.approx(Q * t / (M * H * DT), rel=1e-3)


def test_production_profile_matches_ode():
    t_bottom, depth, t_surf, grad, mcp, rp = 220.0, 1130.0, 25.0, 0.019, 150.0, 1.8
    y = np.linspace(0, depth, 50)
    analytic = production_temperature_profile(t_bottom, y, depth, t_surf, grad, mcp, rp)
    te = lambda yy: t_surf + grad * (depth - yy)  # noqa: E731
    sol = solve_ivp(lambda yy, T: [-(T[0] - te(yy)) / (mcp * rp)], (0, depth), [t_bottom],
                    t_eval=y, rtol=1e-10, atol=1e-10)
    assert np.allclose(analytic, sol.y[0], atol=1e-6)


def test_steam_heat_loss_integral():
    ts, depth, t_surf, grad, rp = 300.0, 1150.0, 25.0, 0.019, 2.0
    num, _ = quad(lambda z: (ts - (t_surf + grad * z)) / rp, 0, depth)
    assert steam_heat_loss_rate(ts, depth, t_surf, grad, rp) == pytest.approx(num, rel=1e-10)
