import mlflow

def init_mlflow():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("OpenRLHF_Training")
