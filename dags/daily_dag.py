from typing import Dict

from airflow.decorators import dag, task
import pendulum

from datasphere import SDK

now = pendulum.now()

@dag(
    dag_id='daily_job',
    start_date=pendulum.datetime(2024, 5, 10, tz="local"),
    schedule=None,#"@daily",
    catchup=False,
)
def run():

    @task(task_id='daily_job')
    def fork_job(source_job_id: str, args: Dict[str, str]):
        sdk = SDK()
        job = sdk.fork_job("id", args={'SCORING_TYPE': "DAILY", "SEGMENT_IDS": "1"}) # user2vec
        job.wait()
        sdk = SDK()
        job = sdk.fork_job("id", args={'SCORING_TYPE': "DAILY", "SEGMENT_IDS": "1"}) # ials
        job.wait()
        sdk = SDK()
        job = sdk.fork_job("id", args={'SCORING_TYPE': "DAILY", "SEGMENT_IDS": "1"}) # item_co_liked
        job.wait()
        sdk = SDK()
        job = sdk.fork_job("id", args={'SCORING_TYPE': "DAILY", "SEGMENT_IDS": "1"}) # candidate_ranking
        job.wait()

    fork_job('id', {'SCORING_DT': now})

run()