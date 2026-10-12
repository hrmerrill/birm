"""Ragged multi-metric training data on a shared daily grid."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricData:
    """Observations for one response metric on the shared grid.

    Parameters
    ----------
    values : np.ndarray
        Observed values $Y^{(j)}$, length ``n_j`` (NaNs dropped).
    idx : np.ndarray
        Integer positions into the shared grid, length ``n_j``, strictly increasing.
    exog : dict[str, np.ndarray]
        Exogenous columns $X^{(j)}$, each length ``n_j`` and aligned 1:1 with ``values``.
    """

    values: np.ndarray
    idx: np.ndarray
    exog: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class Panel:
    """Shared daily grid plus per-metric ragged observations.

    Parameters
    ----------
    grid : pd.DatetimeIndex
        Contiguous daily grid spanning ``min`` to ``max`` over all sources.
    load : np.ndarray
        Training load $w_t$, dense length ``T`` (an unlisted day is ``0.0`` = rest).
    metrics : dict[str, MetricData]
        Response metrics, each with its own values, grid positions, and exog.
    scaling : dict[str, tuple[float, float]]
        ``(center, scale)`` per standardized series, for back-transform. Empty until
        [standardize][birm.data.Panel.standardize] is called.
    """

    grid: pd.DatetimeIndex
    load: np.ndarray
    metrics: dict[str, MetricData]
    scaling: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def T(self) -> int:  # noqa: N802  (T = number of timesteps, ML convention)
        """Number of days on the shared grid.

        Returns
        -------
        int
            Length of ``grid``.
        """
        return len(self.grid)

    def densest(self) -> str:
        r"""Metric with the most observations (auto-pinned to $\gamma \equiv 1$).

        Returns
        -------
        str
            Name of the metric with the largest number of observed days.
        """
        return max(self.metrics, key=lambda c: len(self.metrics[c].idx))

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        date: str,
        metric: str,
        value: str,
        load: str,
        exog: dict[str, list[str]] | None = None,
    ) -> Panel:
        """Build a ``Panel`` from one tidy long table.

        Each row carries a ``(date, metric, value)`` reading; the metric's exog columns ride on
        the same rows. Missingness is implied by absent ``(date, metric)`` rows.

        Parameters
        ----------
        df : pd.DataFrame
            Long-form table with ISO-date, metric-label, and value columns.
        date : str
            Name of the ISO-date column.
        metric : str
            Name of the metric-label column.
        value : str
            Name of the value column.
        load : str
            Metric label that denotes the training load $w_t$.
        exog : dict[str, list[str]], optional
            Per-metric exogenous column names, read off that metric's rows.

        Returns
        -------
        Panel
            Validated panel.
        """
        exog = exog or {}
        dates = _parse_dates(df[date], date)
        labels = df[metric].to_numpy()

        load_mask = labels == load
        load_dates = pd.DatetimeIndex(dates[load_mask])
        load_values = df.loc[load_mask, value].to_numpy(dtype=float)

        records: dict[str, _Record] = {}
        for name in pd.unique(labels):
            if name == load:
                continue
            mask = (labels == name) & df[value].notna().to_numpy()
            cols = {c: df.loc[mask, c].to_numpy(dtype=float) for c in exog.get(name, [])}
            records[name] = _Record(
                dates=pd.DatetimeIndex(dates[mask]),
                values=df.loc[mask, value].to_numpy(dtype=float),
                exog=cols,
            )
        return _assemble(load_dates, load_values, records)

    @classmethod
    def from_sheets(
        cls,
        sheets: dict[str, pd.DataFrame],
        date: str,
        load: str,
        exog: dict[str, list[str]] | None = None,
    ) -> Panel:
        """Build a ``Panel`` from one sheet per metric.

        Each sheet has a ``date`` column, a value column named after the metric, and that metric's
        own exog columns; blank cells mark unobserved days.

        Parameters
        ----------
        sheets : dict[str, pd.DataFrame]
            Mapping of metric name to its wide sheet.
        date : str
            Name of the ISO-date column present in every sheet.
        load : str
            Sheet name that denotes the training load $w_t$.
        exog : dict[str, list[str]], optional
            Per-metric exogenous columns, found within each metric's own sheet.

        Returns
        -------
        Panel
            Validated panel.
        """
        exog = exog or {}
        if load not in sheets:
            raise ValueError(f"load sheet {load!r} not found in sheets")

        load_df = sheets[load]
        load_dates = pd.DatetimeIndex(_parse_dates(load_df[date], date))
        load_values = load_df[load].to_numpy(dtype=float)

        records: dict[str, _Record] = {}
        for name, sheet in sheets.items():
            if name == load:
                continue
            dates = _parse_dates(sheet[date], date)
            mask = sheet[name].notna().to_numpy()
            cols = {c: sheet.loc[mask, c].to_numpy(dtype=float) for c in exog.get(name, [])}
            records[name] = _Record(
                dates=pd.DatetimeIndex(dates[mask]),
                values=sheet.loc[mask, name].to_numpy(dtype=float),
                exog=cols,
            )
        return _assemble(load_dates, load_values, records)

    def standardize(self) -> Panel:
        """Center/scale each response and its exog, rescale load to average-day units.

        Each response metric's values and exog columns are z-scored (center/scale stored in
        ``scaling`` for back-transform); load is rescaled by its own mean so gains stay
        $O(1)$.

        Returns
        -------
        Panel
            A new standardized panel.
        """
        scaling: dict[str, tuple[float, float]] = {}

        load_mean = float(self.load.mean())
        load_scale = load_mean if load_mean != 0.0 else 1.0
        scaling["load"] = (0.0, load_scale)
        load = self.load / load_scale

        metrics: dict[str, MetricData] = {}
        for name, ch in self.metrics.items():
            center, scale = _center_scale(ch.values)
            scaling[name] = (center, scale)
            new_exog = {}
            for col, arr in ch.exog.items():
                c, s = _center_scale(arr)
                scaling[f"{name}.{col}"] = (c, s)
                new_exog[col] = (arr - c) / s
            metrics[name] = MetricData((ch.values - center) / scale, ch.idx, new_exog)

        # note: scaling must reach forecast time. Forecasts, simulations and tourneys need it.
        # apply stored scaling params at rollout.
        return Panel(self.grid, load, metrics, scaling)


@dataclass(frozen=True)
class _Record:
    """Per-metric tidy records shared by both constructors.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Observation dates (NaN values already dropped).
    values : np.ndarray
        Observed values aligned with ``dates``.
    exog : dict[str, np.ndarray]
        Exogenous columns aligned with ``dates``.
    """

    dates: pd.DatetimeIndex
    values: np.ndarray
    exog: dict[str, np.ndarray]


def _parse_dates(values: pd.Series, column: str) -> pd.DatetimeIndex:
    """Parse a date column, raising a user-friendly error on unparseable values.

    Parameters
    ----------
    values : pd.Series
        Raw date column to parse.
    column : str
        Column name, used in the error message.

    Returns
    -------
    pd.DatetimeIndex
        Parsed dates.

    Raises
    ------
    ValueError
        If any value cannot be parsed as a date, naming the offending entries.
    """
    parsed = pd.to_datetime(values, errors="coerce")
    bad = parsed.isna() & values.notna()
    if bad.any():
        offenders = values[bad].unique().tolist()
        raise ValueError(
            f"could not parse dates in column {column!r}: {offenders}. "
            "Use ISO dates like '2025-01-03'."
        )
    return pd.DatetimeIndex(parsed)


def _center_scale(arr: np.ndarray) -> tuple[float, float]:
    """Return ``(mean, std)`` for z-scoring, guarding a zero/degenerate std.

    Parameters
    ----------
    arr : np.ndarray
        Values to summarize.

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation (the latter forced to ``1.0`` when non-positive).
    """
    center = float(np.mean(arr))
    scale = float(np.std(arr))
    return center, scale if scale > 0.0 else 1.0


def _assemble(
    load_dates: pd.DatetimeIndex,
    load_values: np.ndarray,
    records: dict[str, _Record],
) -> Panel:
    """Build the union grid, densify load, and align each metric onto the grid.

    Parameters
    ----------
    load_dates : pd.DatetimeIndex
        Dates on which load was recorded.
    load_values : np.ndarray
        Load values aligned with ``load_dates``.
    records : dict[str, _Record]
        Per-metric tidy records to place on the shared grid.

    Returns
    -------
    Panel
        Validated panel.

    Raises
    ------
    ValueError
        If no dates are present, a load value is negative, or a metric has duplicate observation
        dates.
    """
    all_dates = [load_dates] + [r.dates for r in records.values()]
    stacked = pd.DatetimeIndex(np.concatenate([d.values for d in all_dates])) if all_dates else None
    if stacked is None or len(stacked) == 0:
        raise ValueError("no dates found across load and metrics")
    if (load_values < 0).any():
        raise ValueError(
            "training load values must be >= 0 (a missing day is treated as rest = 0); "
            f"found negative values: {load_values[load_values < 0].tolist()}"
        )
    grid = pd.date_range(stacked.min(), stacked.max(), freq="D")

    load = pd.Series(load_values, index=load_dates).reindex(grid, fill_value=0.0).to_numpy(float)

    metrics: dict[str, MetricData] = {}
    for name, rec in records.items():
        order = np.argsort(rec.dates.values, kind="stable")
        idx = grid.get_indexer(rec.dates[order])
        if (idx < 0).any():
            raise ValueError(f"metric {name!r} has dates outside the grid")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"metric {name!r} has duplicate observation dates")
        exog = {c: arr[order] for c, arr in rec.exog.items()}
        metrics[name] = MetricData(rec.values[order], idx, exog)

    return Panel(grid, load, metrics)
