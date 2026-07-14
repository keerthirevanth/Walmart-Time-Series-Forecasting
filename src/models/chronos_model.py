"""Chronos time-series foundation model forecaster.

Chronos is a pretrained probabilistic forecasting model that treats a numeric
series like a language and generates future values. It is used here zero-shot:
no training on M5 at all, only the historical context of each series is fed in at
inference. This measures what a general pretrained model buys against a model
tuned specifically for this data (LightGBM).

The forecaster keeps the same call signature as the baselines so it runs through
the shared rolling-origin backtest unchanged. It also exposes quantile forecasts
so the probabilistic pinball metric can be computed later.

Heavy dependencies (torch, chronos-forecasting) are imported lazily, so importing
this module does not require them until a forecaster is actually constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation.metrics import ID_COLUMNS


def _horizon_columns(horizon: int) -> list[str]:
    return [f"h_{i}" for i in range(1, horizon + 1)]


@dataclass
class ChronosForecaster:
    cfg: Config
    model_name: str | None = None
    device: str | None = None
    batch_size: int = 256
    quantile_levels: list[float] | None = None
    # Point forecast used for WRMSSE/MASE/RMSE. For intermittent demand the
    # median of the predictive distribution is biased toward zero and badly
    # under-forecasts aggregates, so the distribution mean is the better default.
    point_estimate: str = "mean"
    name: str = "chronos_zeroshot"
    _pipeline: Any = field(default=None, repr=False)
    _torch: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import torch
        from chronos import BaseChronosPipeline

        self._torch = torch
        if self.model_name is None:
            self.model_name = self.cfg.get(
                "models.foundation.model_name", "amazon/chronos-t5-small"
            )
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.quantile_levels is None:
            self.quantile_levels = list(self.cfg.get("evaluation.quantiles"))

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._pipeline = BaseChronosPipeline.from_pretrained(
            self.model_name, device_map=self.device, torch_dtype=dtype
        )

    def _context_matrix(self, train_wide: pd.DataFrame, train_cols: list[str]) -> np.ndarray:
        context_length = int(self.cfg.get("models.foundation.context_length", 512))
        ctx_cols = train_cols[-context_length:]
        return train_wide[ctx_cols].to_numpy(dtype=np.float32)

    def predict_quantiles(
        self, train_wide: pd.DataFrame, train_cols: list[str], horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (point, quantiles) arrays.

        point has shape (n_series, horizon) and is the mean or median of the
        predictive distribution depending on ``self.point_estimate``. quantiles
        has shape (n_series, horizon, n_quantile_levels).
        """
        torch = self._torch
        data = self._context_matrix(train_wide, train_cols)
        n = data.shape[0]

        point_out = np.empty((n, horizon), dtype=np.float32)
        quantile_out = np.empty((n, horizon, len(self.quantile_levels)), dtype=np.float32)
        mid = self.quantile_levels.index(0.5)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            contexts = [torch.tensor(row) for row in data[start:end]]
            # The first argument was renamed from `context` to `inputs` in
            # chronos-forecasting 2.x; pass it positionally to support both.
            q, mean = self._pipeline.predict_quantiles(
                contexts,
                prediction_length=horizon,
                quantile_levels=self.quantile_levels,
            )
            q_np = q.float().cpu().numpy()  # (batch, horizon, n_levels)
            quantile_out[start:end] = q_np
            if self.point_estimate == "mean":
                point_out[start:end] = mean.float().cpu().numpy()
            else:
                point_out[start:end] = q_np[:, :, mid]

        point_out = np.clip(point_out, 0, None)
        quantile_out = np.clip(quantile_out, 0, None)
        return point_out, quantile_out

    def __call__(self, train_wide, train_cols, horizon):
        point, _ = self.predict_quantiles(train_wide, train_cols, horizon)
        out = train_wide[ID_COLUMNS].copy().reset_index(drop=True)
        out[_horizon_columns(horizon)] = point
        return out
