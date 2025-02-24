import mlflow
from ingest_data import ingest_data
from train import train_model
from score import score_model


def main():
    remote_server_uri = "http://0.0.0.0:5000" # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)

    exp_name = "HousePricing_Predictor"
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="full_workflow_run") as parent_run:
        mlflow.log_param("workflow", "House Pricing Predictor")

        with mlflow.start_run(run_name="data_ingestion", nested=True):
            ingest_data("/home/runner/work/mle-training/mle-training/processed_data")
            print("Save to: {}".format(mlflow.get_artifact_uri()))

        with mlflow.start_run(run_name="model_training", nested=True):
            train_model("/home/runner/work/mle-training/mle-training/processed_data", "/home/runner/work/mle-training/mle-training/models")
            print("Save to: {}".format(mlflow.get_artifact_uri()))

        with mlflow.start_run(run_name="model_scoring", nested=True):
            score_model("/home/runner/work/mle-training/mle-training/processed_data", "/home/runner/work/mle-training/mle-training/models")
            print("Save to: {}".format(mlflow.get_artifact_uri()))

if __name__ == "__main__":
    main()
