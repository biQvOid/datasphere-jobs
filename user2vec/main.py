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

    item_embeddings = np.load(f"{PATH}/user2vec/item_embs.npy")

    with open(f'{PATH}/user2vec/item2id.pickle', 'rb') as f:
        item2id = pickle.load(f)

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
        .filter(pl.col("rn") <= 200)
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

    alpha = 0.9

    def calc_uservec(row):
        ids = [item2id[item["item_id"]] for item in row if item["item_id"] in item2id]
        user_item_embs = item_embeddings[ids]
        weights = np.reshape(alpha ** np.arange(len(ids)), (len(ids), 1))
        user_vec = np.sum(user_item_embs * weights, axis=0)
        norm = np.linalg.norm(user_vec)
        return (user_vec / norm).tolist()
    
    seq_data = (
        user_events
        .groupby("user_id")
        .agg(
            pl.struct(["item_id", "stime"])
            .sort_by("stime", descending=True)
            .alias("events"),
        )
        .with_columns(
            pl.col("events").apply(calc_uservec).alias("user_vec")
        )
        .select("user_id", "user_vec")
    )

    print("User vectors successfully calculated")

    user_ids = seq_data["user_id"].to_list()
    user_vectors = seq_data["user_vec"].to_list()

    index = faiss.read_index(f"{PATH}/item2item/item_index_tiny.faiss")

    with open(f"{PATH}/item2item/item_ids", "rb") as fp:
        item_ids = pickle.load(fp)

    batch_size = 1024
    user_recs = {}

    replace_func = np.vectorize(lambda x: item_ids[x])

    for i in tqdm(range(0, len(user_ids), batch_size)):
        cur_vectors = np.array(user_vectors[i:i+batch_size])
        cur_ids = user_ids[i:i+batch_size]
        _, idx = index.search(cur_vectors, k=300)
        recs = replace_func(idx)
        cur_recs = {cur_ids[i]: recs[i, :].tolist() for i in range(len(cur_ids))}
        user_recs = {**user_recs, **cur_recs}
    
    print("Recs successfully calculated")

    recs = (
        seq_data
        .with_columns(
            pl.col("user_id").apply(lambda x: user_recs[x]).alias("recs")
        )
    )

    recs.select("user_id", "recs").write_parquet(f"{PATH}/user2vec/recs.parquet")
    print("Recs successfully saved")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.scoring_dt, args.segment_ids, args.scoring_type)