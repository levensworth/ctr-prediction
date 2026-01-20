
from src.api.dtos import PredictionInput, PredictionOutput
from src.pipelines.prediction import PredictionPipeline


class CTRPredictionService:
    """Service class that encapsulates the prediction pipeline."""

    def __init__(
        self, pipeline: PredictionPipeline, feature_set_name: str = "test_features"
    ) -> None:
        self._pipeline = pipeline
        self._feature_set_name = feature_set_name

    def predict_single(self, input_data: PredictionInput) -> PredictionOutput:
        """Generate prediction for a single publication-campaign pair."""
        publication_id = str(input_data.publication_id)
        campaign_id = str(input_data.campaign_id)

        result, _, _ = self._pipeline.predict_with_imputation(
            publication_id=publication_id,
            campaign_id=campaign_id,
        )

        return PredictionOutput(
            estimated_ctr=result.predicted_ctr,
            model_id=result.model_version or "unknown",
        )

    def predict_batch(self, inputs: list[PredictionInput]) -> list[PredictionOutput]:
        """Generate predictions for multiple publication-campaign pairs."""
        requests = [(str(input.publication_id), str(input.campaign_id)) for input in inputs]
        predictions = self._pipeline.predict_batch_with_imputation(requests)
        return [PredictionOutput(
            estimated_ctr=prediction.predicted_ctr,
            model_id=prediction.model_version or "unknown",
        ) for prediction, _, _ in predictions]

