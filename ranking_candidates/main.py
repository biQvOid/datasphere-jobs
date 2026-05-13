import os
import polars as pl
import argparse
import xxhash
import pytz
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
from catboost import CatBoostRanker
import catboost as cb
import numpy as np
import pandas as pd
import pickle
from utils import filter_aggregation
import pendulum

parser = argparse.ArgumentParser()
parser.add_argument('--scoring_dt', required=True, help='scoring date')
parser.add_argument('--segment_ids', default=100, type=int, required=True, help='user hash ids')
parser.add_argument('--scoring_type', default="DAILY", required=True, help='scoring type (dayly or NRT)')

def calc_user_segment(user_id):
    return abs(xxhash.xxh64(str(user_id), seed=42).intdigest()) % 100

def main(scoring_dt, segment_ids, scoring_type):
    PATH = os.environ["DS_PROJECT_HOME"]
    load_dotenv()
    SCORING_DT = pd.to_datetime(scoring_dt)
    
    candidates_paths = {
        "item_co_occurance_recs": f"{PATH}/item_co_occurance/recs.parquet",
        "user2vec_recs": f"{PATH}/user2vec/recs.parquet",
        "ials_recs": f"{PATH}/ials/recs.parquet"
    }

    if scoring_type == "NRT":
        with open(f"{PATH}/scoring_dt", "rb") as fp:
            scoring_dt = pickle.load(fp)

        now = pendulum.parse(scoring_dt)
        print(f"SCORING_DT: {scoring_dt}")
        user_events = (
            pl.read_parquet(f"{PATH}/user_events.parquet")
            .filter(pl.col("stime") < now)
            .with_columns(pl.col("stime").cast(pl.Date).alias("date"))
        )
    else:
        SCORING_DT = pd.to_datetime(scoring_dt)
        print(f"SCORING_DT: {scoring_dt}")
        user_events = (
            pl.read_parquet(f"{PATH}/user_events.parquet")
            .with_columns(pl.col("stime").cast(pl.Date).alias("date"))
            .filter(pl.col("date") < SCORING_DT)
        )

    user_events = (
        user_events
        .with_columns(
            pl.col("c1_name").fill_null("NO_INFO"),
            pl.col("c2_name").fill_null("NO_INFO"),
            pl.col("brand_name").fill_null("NO_INFO"),
            pl.col("item_condition_name").fill_null("NO_INFO"),
            pl.col("size_name").fill_null("NO_INFO"),
            pl.col("color").fill_null("NO_INFO")
        )
    )

    user_ids = set()
    candidate_sources = list(candidates_paths.keys())

    for key in candidates_paths:
        candidates = pl.read_parquet(candidates_paths[key])
        user_ids = user_ids | set(candidates["user_id"].to_list())
    
    user_segment = pl.DataFrame(list(set(user_ids)), schema=["user_id"])

    if scoring_type == "NRT":
        user_segment = (
            user_segment
            .join(
                pl.read_parquet(f"{PATH}/user_segment.parquet"),
                on="user_id",
                how="inner"
            )
        )

    print("User segment for scoring: ", user_segment.shape[0])
    empty_recs = pl.Series([[]] * len(user_segment), dtype=pl.List(pl.Int64))

    all_candidates = (
        user_segment
        .join(
            pl.read_parquet(candidates_paths[candidate_sources[0]])
            .rename({"recs": candidate_sources[0]}),
            on="user_id",
            how="left"
        )
        .with_columns(
            pl.when(pl.col(candidate_sources[0]).is_null())
            .then(empty_recs)
            .otherwise(pl.col(candidate_sources[0]))
            .alias(candidate_sources[0])
        )
    )

    print(f"Joined {candidate_sources[0]} candidates")

    for i in range(1, len(candidate_sources)):
        all_candidates = (
            all_candidates
            .join(
                pl.read_parquet(candidates_paths[candidate_sources[i]])
                .rename({"recs": candidate_sources[i]}),
                on="user_id",
                how="left"
            )
            .with_columns(
                pl.when(pl.col(candidate_sources[i]).is_null())
                .then(empty_recs)
                .otherwise(pl.col(candidate_sources[i]))
                .alias(candidate_sources[i])
            )
        )

        print(f"Joined {candidate_sources[i]} candidates")

    def join_recs(row):
        joined_recs = set()
        for key in row:
            joined_recs = joined_recs | set(row[key])
        return list(joined_recs)
    
    all_candidates = (
        all_candidates
        .with_columns(
            pl.struct([*candidate_sources]).apply(join_recs).alias("recs")
        )
        .explode("recs")
        .select(
            pl.col("user_id"),
            pl.col("recs").alias("item_id")
        )
        .filter(pl.col("item_id").is_not_null())
    )

    print("All candidates null count: ", all_candidates.filter(pl.col("item_id").is_null()).shape)

    print("Rows count for scoring:", all_candidates.shape[0])

    event_types = [
        'buy_comp', 'item_add_to_cart_tap', 'item_like',
        'buy_start', 'offer_make', 'item_view'
    ]

    user_feats = (
        user_events
        .join(
            all_candidates
            .select("user_id")
            .unique(),
            on="user_id",
            how="inner"
        )
        .groupby("user_id")
        .agg(
            filter_aggregation("price", filter_by=("event_id", "buy_comp"), agg="sum"),
            filter_aggregation("c0_name", agg="distinct"),
            filter_aggregation("c1_name", agg="distinct"),
            filter_aggregation("c2_name", agg="distinct"),
            filter_aggregation("brand_name", agg="distinct"),
            filter_aggregation("product_id", agg="distinct"),
            *[filter_aggregation("brand_name", filter_by=("event_id", event_type), agg="last") for event_type in event_types],
            *[filter_aggregation("c0_name", filter_by=("event_id", event_type), agg="last") for event_type in event_types],
            *[filter_aggregation("c1_name", filter_by=("event_id", event_type), agg="last") for event_type in event_types],
            *[filter_aggregation("c2_name", filter_by=("event_id", event_type), agg="last") for event_type in event_types],
            *[filter_aggregation("product_id", filter_by=("event_id", event_type), agg="distinct") for event_type in event_types],
            *[filter_aggregation("brand_name", filter_by=("event_id", event_type), agg="distinct") for event_type in event_types],
            *[filter_aggregation("c0_name", filter_by=("event_id", event_type), agg="distinct") for event_type in event_types],
            *[filter_aggregation("c1_name", filter_by=("event_id", event_type), agg="distinct") for event_type in event_types],
            *[filter_aggregation("c2_name", filter_by=("event_id", event_type), agg="distinct") for event_type in event_types],
            *[filter_aggregation("price", filter_by=("event_id", event_type), agg="mean") for event_type in event_types],
            *[filter_aggregation("price", filter_by=("event_id", event_type), agg="max") for event_type in event_types],
            *[filter_aggregation("price", filter_by=("event_id", event_type), agg="min") for event_type in event_types],
            *[filter_aggregation("price", filter_by=("event_id", event_type), agg="std") for event_type in event_types],
            *[filter_aggregation("item_id", filter_by=("event_id", event_type), agg="cnt") for event_type in event_types],
            *[filter_aggregation("item_id", filter_by=("event_id", event_type), agg="distinct") for event_type in event_types],
            *[filter_aggregation("price", filter_by=("event_id", event_type), agg="mean", scoring_dt=SCORING_DT) for event_type in event_types],
            *[filter_aggregation("price", filter_by=("event_id", event_type), agg="max", scoring_dt=SCORING_DT) for event_type in event_types],
            *[filter_aggregation("price", filter_by=("event_id", event_type), agg="min", scoring_dt=SCORING_DT) for event_type in event_types],
            *[filter_aggregation("price", filter_by=("event_id", event_type), agg="std", scoring_dt=SCORING_DT) for event_type in event_types],
            *[filter_aggregation("item_id", filter_by=("event_id", event_type), agg="cnt", scoring_dt=SCORING_DT) for event_type in event_types],
            *[filter_aggregation("item_id", filter_by=("event_id", event_type), agg="distinct", scoring_dt=SCORING_DT) for event_type in event_types]
        )
    )

    print("User feats shape: ", user_feats.shape)

    item_properties = [
        "c0_name", "c1_name", "c2_name", "brand_name",
        "item_condition_name", "size_name", "color", "price"
    ]

    selected_event_types = [
        "item_add_to_cart_tap", "item_view", "item_like"
    ]

    item_feats = (
        user_events
        .join(
            all_candidates
            .select("item_id")
            .unique(),
            on="item_id",
            how="inner"
        )
        .groupby("item_id")
        .agg(
            *[pl.first(col_name) for col_name in item_properties],
            *[filter_aggregation("user_id", filter_by=("event_id", event_type), agg="cnt") for event_type in selected_event_types],
            *[filter_aggregation("user_id", filter_by=("event_id", event_type), agg="distinct") for event_type in selected_event_types],
            *[filter_aggregation("user_id", filter_by=("event_id", event_type), agg="cnt", scoring_dt=SCORING_DT) for event_type in selected_event_types],
            *[filter_aggregation("user_id", filter_by=("event_id", event_type), agg="distinct", scoring_dt=SCORING_DT) for event_type in selected_event_types]
        )
    )

    print("Item feats shape: ", item_feats.shape)

    join_on = {}

    for col_name in ["c0_name", "c1_name", "c2_name", "brand_name", "item_condition_name", "size_name", "color"]:
        unique_col_values = (
            user_events
            .join(
                all_candidates
                .select("item_id")
                .unique(),
                on="item_id",
                how="inner"
            )
            .select(col_name)
            .unique()
        )
    
        feats = (
            user_events
            .join(
                all_candidates
                .select("user_id")
                .unique(),
                on="user_id",
                how="inner"
            )
            .join(
                unique_col_values,
                on=col_name,
                how="inner",
            )
            .groupby("user_id", col_name)
            .agg(
                *[filter_aggregation("price", filter_by=("event_id", event_type), agg="mean", suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("price", filter_by=("event_id", event_type), agg="max", suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("price", filter_by=("event_id", event_type), agg="min", suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("price", filter_by=("event_id", event_type), agg="std", suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("item_id", filter_by=("event_id", event_type), agg="cnt", suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("item_id", filter_by=("event_id", event_type), agg="distinct", suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("price", filter_by=("event_id", event_type), agg="mean", scoring_dt=SCORING_DT, suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("price", filter_by=("event_id", event_type), agg="max", scoring_dt=SCORING_DT, suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("price", filter_by=("event_id", event_type), agg="min", scoring_dt=SCORING_DT, suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("price", filter_by=("event_id", event_type), agg="std", scoring_dt=SCORING_DT, suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("item_id", filter_by=("event_id", event_type), agg="cnt", scoring_dt=SCORING_DT, suffix=f"by={col_name}") for event_type in event_types],
                *[filter_aggregation("item_id", filter_by=("event_id", event_type), agg="distinct", scoring_dt=SCORING_DT, suffix=f"by={col_name}") for event_type in event_types]
            )
        )
    
        join_on[col_name] = feats

    print("Finished calculating user-item interactions features")

    all_candidates = (
        all_candidates
        .join(
            user_feats,
            on="user_id",
            how="left"
        )
        .join(
            item_feats,
            on="item_id",
            how="left",
        )
    )

    for col_name in join_on:
        all_candidates = all_candidates.join(join_on[col_name], on=[col_name, "user_id"], how="left")

    print("Final dataframe shape: ", all_candidates.shape)

    cat_features = all_candidates.select(pl.col(pl.Utf8)).columns
    features = [col_name for col_name in all_candidates.columns if col_name not in ["user_id", "item_id"]]

    print(all_candidates.select("user_id", "item_id").head(10))

    all_candidates = all_candidates.to_pandas()

    print(all_candidates[["user_id", "item_id"]].head(10))

    all_candidates[cat_features] = all_candidates[cat_features].fillna("MISSING")

    candidates_pool = cb.Pool(
        all_candidates[features],
        cat_features=cat_features
    )

    print("Inference catboost pool prepared")

    model = CatBoostRanker()
    model.load_model(f'{PATH}/models/ranker_v1.cbm')

    print("Model loaded")

    all_candidates["score"] = model.predict(candidates_pool)

    print("Scoring finished")

    #print(all_candidates[["user_id", "item_id", "score"]].head(5))

    #all_candidates.to_parquet(f"{PATH}/scores.parquet")

    item_attributes = (
        pl.read_parquet(f"{PATH}/user_events.parquet")
        .with_columns(pl.col("stime").cast(pl.Date).alias("date"))
        .filter(pl.col("date") <= pd.to_datetime("2023-05-21"))
        .select("item_id", "name", "price")
        .unique()
    )

    all_candidates = pl.from_pandas(all_candidates)

    all_candidates = (
        all_candidates
        .join(
            item_attributes,
            on="item_id",
            how="left"
        )
        .with_columns(
            pl.lit(scoring_dt).alias("scoring_dt")
        )
        .select(
            "user_id", "item_id", "name",
            "price", "score", "scoring_dt"
        )
    )

    invalid_items = (
        pl.read_parquet(f"{PATH}/user_events.parquet")
        .filter(pl.col("event_id") == "buy_comp")
        .with_columns(pl.lit(1).alias("is_bought"))
        .select("item_id", "is_bought")
    )

    all_candidates = (
        all_candidates
        .join(
            invalid_items,
            on="item_id",
            how="left"
        )
        .filter(pl.col("is_bought").is_null())
        .select(
            "user_id", "item_id", "name",
            "price", "score", "scoring_dt"
        )
        .to_pandas()
    )

    dbname = os.environ.get("DB_NAME", "postgres")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASS", "mysecretpassword")
    host = os.environ.get("DB_HOST", "db")
    port = os.environ.get("DB_PORT", "5432")

    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
    all_candidates.to_sql('model_scores_v2', engine, if_exists='append', index=False)

    print("Scored saved in db")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.scoring_dt, args.segment_ids, args.scoring_type)