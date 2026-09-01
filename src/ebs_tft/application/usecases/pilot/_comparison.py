"""Build paired cumulative-depth diagnostics from pilot metrics."""

from __future__ import annotations

import polars as pl

_NEURAL_MODELS = ("deeplob_direction", "tft_direction")
_METRICS = ("balanced_accuracy", "macro_f1", "mcc", "log_loss")


def cumulative_depth_comparison(*, metrics: pl.DataFrame) -> pl.DataFrame:
    """Compare every available cumulative depth with its preceding depth."""
    neural = metrics.filter(pl.col("model").is_in(_NEURAL_MODELS))
    depths = sorted(neural["depth"].unique().to_list())
    identity_candidates = (
        "instrument",
        "trading_date",
        "horizon_steps",
        "horizon_milliseconds",
        "training_mode",
        "model",
        "seed",
    )
    keys = [name for name in identity_candidates if name in neural.columns]
    comparisons: list[pl.DataFrame] = []
    for shallower_depth, deeper_depth in zip(depths, depths[1:], strict=False):
        shallower = neural.filter(pl.col("depth") == shallower_depth).select(
            [
                *keys,
                *(pl.col(name).alias(f"{name}_shallower") for name in _METRICS),
            ]
        )
        deeper = neural.filter(pl.col("depth") == deeper_depth).select(
            [
                *keys,
                *(pl.col(name).alias(f"{name}_deeper") for name in _METRICS),
            ]
        )
        comparisons.append(
            shallower.join(deeper, on=keys, how="inner", validate="1:1")
            .with_columns(
                pl.lit(shallower_depth).alias("shallower_depth"),
                pl.lit(deeper_depth).alias("deeper_depth"),
                *(
                    (pl.col(f"{name}_deeper") - pl.col(f"{name}_shallower")).alias(
                        f"{name}_delta"
                    )
                    for name in _METRICS
                ),
            )
            .select(
                [
                    *keys,
                    "shallower_depth",
                    "deeper_depth",
                    *(
                        column
                        for name in _METRICS
                        for column in (
                            f"{name}_shallower",
                            f"{name}_deeper",
                            f"{name}_delta",
                        )
                    ),
                ]
            )
        )
    if comparisons:
        return pl.concat(comparisons).sort([*keys, "shallower_depth", "deeper_depth"])
    return pl.DataFrame(
        schema={
            **{name: neural.schema[name] for name in keys},
            "shallower_depth": pl.Int64,
            "deeper_depth": pl.Int64,
            **{
                column: pl.Float64
                for name in _METRICS
                for column in (
                    f"{name}_shallower",
                    f"{name}_deeper",
                    f"{name}_delta",
                )
            },
        }
    )
