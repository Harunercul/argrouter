"""thriftllm — LLM cost routing with a correct per-request cost.

Public API:
    expected_cost, Catalog, ModelPrice         — the cost model
    select, Endpoint, Policy                   — provider selection for one model
"""

from thriftllm.catalog.pricing import (
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
from thriftllm.forecast import Forecast, OutputForecaster
from thriftllm.providers import (
    Endpoint,
    NoEligibleProvider,
    Policy,
    Selection,
    select,
)

__version__ = "0.1.0.dev0"

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
