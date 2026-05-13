import os
import polars as pl
import argparse
import xxhash
from tqdm import tqdm
import pytz
from datetime import datetime, timedelta
import pendulum
import pickle
from scipy.sparse import coo_array
import scipy
import implicit
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
    
    event_weights = {
        "item_view": 1,
        "item_like": 1,
        "item_add_to_cart_tap": 1,
        "offer_make": 1,
        "buy_start": 1,
        "buy_comp": 1
    }

    train_interactions = (
        user_events
        .with_columns(
            pl.col("user_id").apply(calc_user_segment).alias("user_segment")
        )
        .filter(pl.col("user_segment").is_in(list(range(segment_ids))))
        .with_columns(pl.col("event_id").apply(lambda x: event_weights[x]).alias("event_weight"))
    )

    all_users = train_interactions.select("user_id").unique()

    agg_train_interactions = (
        train_interactions
        .groupby("user_id", "item_id")
        .agg(pl.sum("event_weight").alias("interaction_weight"))
    )

    def df_to_matrix(df, weight=400):
        item2id = {item: i for i, item in enumerate(train_interactions["item_id"].unique())}
        user2id = {user: i for i, user in enumerate(train_interactions["user_id"].unique())}
        id2user = {i: user for i, user in enumerate(train_interactions["user_id"].unique())}
        users_cnt = len(user2id)
        items_cnt = len(item2id)
        pandas_df = df.select("user_id", "item_id", "interaction_weight").unique().to_pandas()
        pandas_df["item_id"] = pandas_df["item_id"].apply(lambda x: item2id[x])
        pandas_df["user_id"] = pandas_df["user_id"].apply(lambda x: user2id[x])
        result = coo_array((pandas_df["interaction_weight"] * weight, (pandas_df["user_id"], pandas_df["item_id"])), shape=(users_cnt, items_cnt))
        return result, item2id, user2id
    
    interaction_matrix, item2id, user2id = df_to_matrix(agg_train_interactions)
    sparse_interactions = scipy.sparse.csr_matrix(interaction_matrix)
    id2user = {v: k for k, v in user2id.items()}
    id2item = {id_: item for item, id_ in item2id.items()}

    print("Interaction matrix built")

    model = implicit.als.AlternatingLeastSquares(
        factors=128, iterations=30, regularization=0.001, # 456
        calculate_training_loss=True, random_state=42
    )

    model.fit(sparse_interactions)

    print("iALS train completed")

    als_recs = model.recommend(
        range(sparse_interactions.shape[0]),
        sparse_interactions, N=400, filter_already_liked_items=True
    )[0]

    def get_recs(row):
        if row["user_id"] in user2id:
            als_ids = als_recs[user2id[row["user_id"]]]
            recs = [int(id2item[id_]) for id_ in als_ids]
            return recs
        return []

    if scoring_type == "NRT":
        all_users = (
            all_users
            .join(
                pl.read_parquet(f"{PATH}/user_segment.parquet"),
                on="user_id",
                how="inner"
            )
        )

    recs = (
        all_users
        .with_columns(
            pl.struct(["user_id"]).apply(get_recs).alias("recs")
        )
    )

    recs.select("user_id", "recs").write_parquet(f"{PATH}/ials/recs.parquet")
    print("Recs successfully saved")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.scoring_dt, args.segment_ids, args.scoring_type)