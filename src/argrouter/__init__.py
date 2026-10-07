"""argrouter — LLM cost routing with a correct per-request cost.

Public API:
    expected_cost, Catalog, ModelPrice         — the cost model
    select, Endpoint, Policy                   — provider selection for one model
"""

from argrouter.catalog.pricing import (
    CacheRates,
    Catalog,
    ContextTier,
    CostBreakdown,
    ModelPrice,
    PricingError,
    UnpricedModel,
    cost_per_quality_point,
    expected_cost,
)
from argrouter.forecast import Forecast, OutputForecaster
from argrouter.providers import (
    Endpoint,
    NoEligibleProvider,
    Policy,
    Selection,
    select,
)

__version__ = "0.1.0a1"

__all__ = [
    "CacheRates",
    "Catalog",
    "ContextTier",
    "CostBreakdown",
    "Endpoint",
    "Forecast",
    "ModelPrice",
    "NoEligibleProvider",
    "OutputForecaster",
    "Policy",
    "PricingError",
    "Selection",
    "UnpricedModel",
    "cost_per_quality_point",
    "expected_cost",
    "select",
]
