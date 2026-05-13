import polars as pl
import pandas as pd
from datetime import timedelta

def filter_aggregation(agg_colname, filter_by=None, agg="cnt", dt_colname="date", scoring_dt=None, day_wnd=7, suffix=None):
    base_expr = pl.col(agg_colname)
    
    if filter_by:
        filter_colname, filter_value = filter_by
        base_expr = base_expr.filter(pl.col(filter_colname) == filter_value)
        
    if dt_colname and scoring_dt:
        start_dt = pd.to_datetime(scoring_dt) + timedelta(days=-day_wnd)
        end_dt = pd.to_datetime(scoring_dt)

        base_expr = (
            base_expr
            .filter(pl.col(dt_colname) >= start_dt)
            .filter(pl.col(dt_colname) < end_dt)
        )

    if agg == "last" or agg == "first":
        base_expr = base_expr.sort_by("stime", descending=False)
        
    agg_map = {
        "cnt": base_expr.count,
        "distinct": base_expr.n_unique,
        "max": base_expr.max,
        "mean": base_expr.mean,
        "min": base_expr.min,
        "std": base_expr.std,
        "sum": base_expr.sum,
        "last": base_expr.last,
        "first": base_expr.first
    }
    
    if agg not in agg_map:
        raise ValueError(f"Unknown aggregation: {agg}")
        
    col_name = f"{agg_colname}_{agg}"
    
    col_name = f"{filter_colname}={filter_value}_{col_name}" if filter_by else col_name
    col_name = f"{col_name}_wnd={day_wnd}" if dt_colname and scoring_dt else col_name
    col_name = f"{col_name}_{suffix}" if suffix else col_name

    return (
        agg_map[agg]()
        .alias(col_name)
    )