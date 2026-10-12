"""Tests for the :mod:`birm.core.observation` link, orthogonalization, and mean helpers."""

import numpy as np
import pytensor.tensor as pt
import pytest

from birm.core.observation import (
    apply_inverse_link,
    apply_link,
    observation_mean,
    orthogonalize,
)


@pytest.mark.parametrize(
    ("link", "x"),
    [("identity", 0.3), ("log", 2.0), ("logit", 0.3)],
)
def test_link_roundtrips(link: str, x: float) -> None:
    """Each inverse link undoes its forward link (numpy inputs -> numpy out)."""
    assert apply_inverse_link(apply_link(x, link), link) == pytest.approx(x)


def test_link_values() -> None:
    """The forward links compute their textbook transforms."""
    assert apply_link(2.0, "log") == pytest.approx(np.log(2.0))
    assert apply_link(0.25, "logit") == pytest.approx(np.log(0.25 / 0.75))
    assert apply_link(5.0, "identity") == 5.0


def test_link_symbolic_dispatch() -> None:
    """A symbolic input dispatches through to a PyTensor variable."""
    x = pt.scalar("x")
    out = apply_link(x, "log")
    assert hasattr(out, "eval")
    assert out.eval({x: 2.0}) == pytest.approx(np.log(2.0))


def test_unknown_link_rejected() -> None:
    """An unrecognized link name is an error on both directions."""
    with pytest.raises(ValueError, match="link must be one of"):
        apply_link(1.0, "probit")
    with pytest.raises(ValueError, match="link must be one of"):
        apply_inverse_link(1.0, "probit")


def test_orthogonalize_removes_design_span() -> None:
    """The orthogonalized residual is orthogonal to every design column."""
    rng = np.random.default_rng(0)
    design = rng.normal(size=(50, 3))
    signal = rng.normal(size=50)
    u = orthogonalize(signal, design).eval()
    assert np.allclose(design.T @ u, 0.0, atol=1e-6)


def test_orthogonalize_fixed_point() -> None:
    """A signal already orthogonal to the design passes through unchanged."""
    design = np.eye(6)[:, :2]  # spans the first two coordinates
    signal = np.array([0.0, 0.0, 1.0, -2.0, 3.0, 0.5])  # lives in the complementary span
    u = orthogonalize(signal, design).eval()
    assert np.allclose(u, signal, atol=1e-5)


def test_observation_mean_full() -> None:
    """The mean matches alpha + gamma*l(R) + X@beta + u term by term."""
    rng = np.random.default_rng(1)
    stock = rng.normal(size=8)
    exog = rng.normal(size=(8, 2))
    beta = np.array([0.5, -1.2])
    u = rng.normal(size=8)
    got = observation_mean(
        stock, link="identity", alpha=2.0, gamma=1.5, exog=exog, beta=beta, u=u
    ).eval()
    expected = 2.0 + 1.5 * stock + exog @ beta + u
    assert np.allclose(got, expected)


def test_observation_mean_optional_terms_dropped() -> None:
    """With no exog/u and defaults, the mean is just gamma*l(R) (+0 intercept)."""
    stock = np.array([1.0, 2.0, 0.5])
    got = observation_mean(stock, link="log", gamma=2.0)
    assert np.allclose(got, 2.0 * np.log(stock))


def test_observation_mean_requires_beta_with_exog() -> None:
    """Passing exog without beta (or vice versa) is an error."""
    stock = np.zeros(4)
    with pytest.raises(ValueError, match="exog and beta"):
        observation_mean(stock, exog=np.ones((4, 1)))
    with pytest.raises(ValueError, match="exog and beta"):
        observation_mean(stock, beta=np.ones(1))
