import click
from pipelines.deployment_pipline import (continuous_deployment_pipeline, inference_pipeline)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer)

@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(stop_service: bool):
    """Run the churn classifier deployment pipeline"""
    model_name = "churn_classifier" 

    if stop_service:
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        existing_services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name=model_name,
            running=True,
        )

        if existing_services:
            existing_services[0].stop(timeout=10)
        return

    continuous_deployment_pipeline()

    inference_pipeline()

    print(
        "\n Training, deployment, and inference completed!\n\n"
        "To view experiment tracking, run:\n"
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "Then open http://127.0.0.1:5000 in your browser.\n"
    )

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
    )

    if service and service[0]:
        print(
            f"\n Churn classifier prediction server is running at:\n"
            f"    {service[0].prediction_url}\n"
            f"\n To stop the service, run:\n"
            f"    python run_deployment.py --stop-service"
        )


if __name__ == "__main__":
    run_main()
