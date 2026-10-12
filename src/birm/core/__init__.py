"""Latent-stock cores: the fitness :math:`f_t` and fatigue :math:`g_t` building blocks."""

from birm.core.fatigue import fatigue_stock
from birm.core.fitness import fitness_stock

__all__ = ["fatigue_stock", "fitness_stock"]
