from azure.ai.ml import MLClient, command
from azure.ai.ml.sweep import Choice, Uniform, BanditPolicy
from azure.identity import DefaultAzureCredential


def submit_sweep():
    ml_client = MLClient(DefaultAzureCredential(), "sub_id", "rg_name", "ws_name")

    base_run = command(
        code="./src",  # folder that holds train.py when the job is submitted
        command="python train.py --learning_rate ${{inputs.learning_rate}} --rank ${{inputs.rank}}",
        inputs={"learning_rate": 2e-4, "rank": 8},  # defaults; the sweep overrides them per trial
        compute="gpu-cluster",
    )

    # Task 4: search space. Keys must match the ${{inputs.*}} names in the command string.
    search_space = {
        "learning_rate": Uniform(min_value=1e-5, max_value=1e-4),
        "rank": Choice(values=[8, 16, 32]),
    }

    # Task 5: Bandit early termination. Cancel any trial whose train_loss is more than
    # 10% worse than the best trial, checked every 2 intervals after the first 5.
    early_termination_policy = BanditPolicy(
        slack_factor=0.1,
        evaluation_interval=2,
        delay_evaluation=5,
    )

    sweep_job = base_run.sweep(
        search_space=search_space,
        sampling_algorithm="random",
        primary_metric="train_loss",  # must match the metric name logged in train.py
        goal="Minimize",
        early_termination_policy=early_termination_policy,
    )

    # Hard budget caps on the sweep as a whole.
    sweep_job.set_limits(max_total_trials=20, max_concurrent_trials=4, timeout=7200)

    print(sweep_job)  # prints the YAML Azure ML would receive
    # ml_client.jobs.create_or_update(sweep_job)  # Not required for this activity


if __name__ == "__main__":
    submit_sweep()
