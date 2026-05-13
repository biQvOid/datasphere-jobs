import os
import polars as pl
import argparse
import xxhash
from collections import defaultdict
import pytz
import pendulum
from datetime import datetime, timedelta
from functools import reduce
import pickle
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--scoring_dt', required=True, help='scoring date')
parser.add_argument('--segment_ids', default=100, type=int, required=True, help='user hash ids')
parser.add_argument('--scoring_type', default="DAILY", required=True, help='scoring type (dayly or NRT)')

def calc_user_segment(user_id):
    return abs(xxhash.xxh64(str(user_id), seed=42).intdigest()) % 100

def main(scoring_dt, segment_ids, scoring_type):
    PATH = os.environ["DS_PROJECT_HOME"]
    user_events = pl.read_parquet(f"{PATH}/user_events.parquet")

    SCORING_DT = pd.to_datetime(scoring_dt)

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

    likes = (
        user_events
        .filter(pl.col("event_id") == "item_like")
    )

    sessions_with_likes = (
        likes
        .groupby("session_id", "user_id")
        .agg(pl.n_unique("item_id").alias("likes_cnt"))
        .filter(pl.col("likes_cnt") >= 2)
    )

    session_likes = (
        likes
        .join(
            sessions_with_likes,
            on=["session_id", "user_id"],
            how="inner"
        )
        .select(
            pl.struct("session_id", "user_id").apply(lambda x: f"{x['session_id']}_{x['user_id']}"),
            pl.col("item_id")
        )
        .unique()
    )

    item_occurances = defaultdict(list)

    session_with_item = session_likes.groupby("item_id").agg(pl.col("session_id"))
    session_items = session_likes.groupby("session_id").agg(pl.col("item_id"))
    session_items = dict(zip(session_items["session_id"].to_list(), session_items["item_id"].to_list()))

    item_ids = session_with_item["item_id"].to_list()
    session_ids = session_with_item["session_id"].to_list()

    for item_id, item_sessions in zip(item_ids, session_ids):
        for session in item_sessions:
            co_items = session_items[session]
            for item in co_items:
                if item != item_id:
                    item_occurances[item_id].append(item)

    print("Item co occurance successfully calculated")

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
        .filter(pl.col("rn") <= 150)
        .groupby("user_id")
        .agg(pl.col("item_id").alias("last_events"))
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

    print("Users count: ", user_events.shape[0])

    def get_similar_items(row):
        return list(reduce(lambda x, y: x + y, [item_occurances[item_id] for item_id in row["last_events"] if item_id in item_occurances], []))

    recs = (
        user_events
        .with_columns(
            pl.struct(["last_events"]).apply(get_similar_items).alias("recs")
        )
    )

    recs.select("user_id", "recs").write_parquet(f"{PATH}/item_co_occurance/recs.parquet")
    print("Recs successfully saved")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.scoring_dt, args.segment_ids, args.scoring_type)