import os
import polars as pl
import argparse
import xxhash
import pickle
import pendulum
import pytz
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--scoring_dt', default="None", type=str, help='scoring date')
parser.add_argument('--segment_ids', default=100, type=int, required=True, help='user hash ids')
parser.add_argument('--scoring_timedelta', default=-1, type=int, help='timedelta in minutes for NRT')

def calc_user_segment(user_id):
    return abs(xxhash.xxh64(str(user_id), seed=42).intdigest()) % 100

def main(scoring_dt, segment_ids, scoring_timedelta):
    PATH = os.environ["DS_PROJECT_HOME"]
    user_events = pl.read_parquet(f"{PATH}/user_events.parquet")

    scoring_dt = scoring_dt.replace("_", " ")

    if scoring_dt != "None":
        now = pendulum.parse(scoring_dt)
    else:
        now = pendulum.now(tz="UTC") - timedelta(days=1091)
        #now = datetime.now(pytz.timezone('UTC')) - timedelta(days=1091)

    if scoring_timedelta <= 0:
        print("Scoring timedelta must be positive integer")
        return 
    
    last_events = (
        user_events
        .filter(pl.col("stime") < now)
        .filter(pl.col("stime") >= now - timedelta(minutes=scoring_timedelta))
    )

    print("Time range for selected events:")
    print("Min event dt: ", last_events.select(pl.min("stime"))["stime"][0].strftime("%Y-%m-%d %H:%M:%S"))
    print("Max event dt: ", last_events.select(pl.max("stime"))["stime"][0].strftime("%Y-%m-%d %H:%M:%S"))

    user_segment = (
        last_events
        .select("user_id")
        .unique()
    )

    print("User count for scoring: ", user_segment.shape[0])

    user_segment.write_parquet(f"{PATH}/user_segment.parquet")

    with open(f"{PATH}/scoring_dt", "wb") as fp:
        pickle.dump(now.to_datetime_string(), fp)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.scoring_dt, args.segment_ids, args.scoring_timedelta)