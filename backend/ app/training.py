import ray
import subprocess
import mlflow

@ray.remote(num_gpus=1)
class OpenRLHFTrainer:
    def __init__(self):
        pass

    def train(self, dataset_path, training_method, base_model, job_id):
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.start_run(run_name=job_id)

        cmd = [
            "python",
            "./OpenRLHF/train.py",
            "--dataset_path", dataset_path,
            "--base_model", base_model,
            "--method", training_method,
            "--output_dir", f"./outputs/{job_id}",
            "--deepspeed",
            "--deepspeed_config", "./ds_config.json"
        ]

        process = subprocess.Popen(cmd)
        process.wait()

        mlflow.end_run()
        return "completed"

trainer_actor = OpenRLHFTrainer.remote()

def run_openrlhf_training(dataset_path, training_method, base_model, job_id):
    future = trainer_actor.train.remote(dataset_path, training_method, base_model, job_id)
    return ray.get(future)
