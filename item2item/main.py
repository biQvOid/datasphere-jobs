import os
import polars as pl
import argparse
import xxhash
from tqdm import tqdm
import pytz
from datetime import datetime, timedelta
import pickle
import faiss
import numpy as np
from functools import reduce
import pandas as pd
import pendulum

parser = argparse.ArgumentParser()
parser.add_argument('--scoring_dt', required=True, help='scoring date')
parser.add_argument('--segment_ids', default=100, type=int, required=True, help='user hash ids')
parser.add_argument('--scoring_type', default="DAILY", required=True, help='scoring type (dayly or NRT)')

def calc_user_segment(user_id):
    return abs(xxhash.xxh64(str(user_id), seed=42).intdigest()) % 100

def main(scoring_dt, segment_ids, scoring_type):
    PATH = os.environ["DS_PROJECT_HOME"]
    user_events = pl.read_parquet(f"{PATH}/user_events.parquet")

    index = faiss.read_index(f"{PATH}/complex_item_index_tiny.faiss")

    with open(f"{PATH}/complex_item_ids", "rb") as fp:
        item_ids = pickle.load(fp)

    if scoring_type == "NRT":
        with open(f"{PATH}/scoring_dt", "rb") as fp:
            scoring_dt = pickle.load(fp)

        now = pendulum.parse(scoring_dt)
        print(f"SCORING_DT: {scoring_dt}")
        user_events = (
            user_events
            .filter(pl.col("stime") < now)
        )
    else:
        SCORING_DT = pd.to_datetime(scoring_dt)
        print(f"SCORING_DT: {scoring_dt}")
        user_events = (
            user_events
            .with_columns(pl.col("stime").cast(pl.Date).alias("date"))
            .filter(pl.col("date") < SCORING_DT)
        )

    user_events = (
        user_events
        .with_columns(
            pl.col("user_id").apply(calc_user_segment).alias("user_segment")
        )
        .filter(pl.col("user_segment").is_in(list(range(segment_ids))))
        .select(
            pl.col("user_id"),
            pl.col("item_id"),
            pl.col("stime"),
            pl.col("stime").rank("dense", descending=True).over("user_id").alias("rn")
        )
        .filter(pl.col("rn") <= 40)
    )

    if scoring_type == "NRT":
        user_events = (
            user_events
            .join(
                pl.read_parquet(f"{PATH}/user_segment.parquet"),
                on="user_id",
                how="inner"
            )
        )

    print("User events count: ", user_events.shape[0])

    with open(f"{PATH}/similar_items_lst20_top40_complex_threshold=1.0", "rb") as fp:
        similar_items = pickle.load(fp)

    def get_similar_items(row):
        return list(reduce(lambda x, y: x + y, [similar_items[item_id] for item_id in row["last_clicks"] if item_id in similar_items]))

    recs_20 = (
        user_events
        .with_columns(
            pl.struct(["last_clicks"]).apply(get_similar_items).alias("recs")
        )
)

    recs_20.select("user_id", "recs").write_parquet(f"{PATH}/user2vec/recs.parquet")
    print("Recs successfully saved")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.scoring_dt, args.segment_ids, args.scoring_type)