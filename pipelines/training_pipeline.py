from zenml import Model, pipeline
from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.data_preprocessing_step import data_preprocessing_step
from steps.data_resampling_step import data_resampling_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
import os


@pipeline(
    model=Model(
        name="churn_classifier"
    ),
)
def ml_pipeline():

    """ Define an end-to-end Ml pipeline. """

    # Step 1: Load raw data
    data_path  = os.path.join("data", "Telco-Customer-Churn.csv")
    raw_data = data_ingestion_step(data_path=data_path)

    # Step 2: Split into train/test
    X_train, X_test, y_train, y_test = data_splitter_step(df=raw_data, target_column="Churn")

    # Step 3: Preprocessing
    X_train_prepared, X_test_prepared, y_train_prepared, y_test_prepared = data_preprocessing_step(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    # Step 4: Resample training set
    X_resampled, y_resampled = data_resampling_step(
        X_train_prepared=X_train_prepared, y_train_prepared=y_train_prepared
    )

    # Step 5: Model training
    model = model_building_step(
        X_resampled=X_resampled, y_resampled=y_resampled
    )

    # Step 6: Model evaluation
    evaluation_metrics = model_evaluator_step(
        model=model, X_test=X_test_prepared, y_test=y_test_prepared
    )

    return model, evaluation_metrics


if __name__ == "__main__":
    run = ml_pipeline()

