from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
#from zenml.models import ModelVersion
from zenml.client import Client


@step(enable_cache=False)
def prediction_service_loader(model_name: str, pipeline_name: str, step_name: str) -> MLFlowDeploymentService:
    """Loads an existing MLflow prediction service from a previous deployment."""

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    client = Client()
    try:
        prod_model_version = client.get_model_version(
            model_name="churn_classifier",
            version="production"
        )
        print(f" Production model version found: {prod_model_version.version}")
    except Exception:
        print(f" No production version found for model '{model_name}'")

    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name="churn_classifier"
    )

    if not existing_services:
        raise RuntimeError(
            f" No MLflow prediction service deployed by the "
            f"{step_name} step in the {pipeline_name} pipeline is currently running."
        )

    service = existing_services[0]

    if not service.is_running:
        raise RuntimeError(f" The prediction service {service} is not running!")

    print(f" Loaded prediction service: {service}")

    return service