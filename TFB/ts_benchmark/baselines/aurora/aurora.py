from ts_benchmark.baselines.aurora.modeling_aurora import AuroraForPrediction
from ts_benchmark.baselines.foundation_forecasting_model_base import FoundationForecastingModelBase
from einops import rearrange

# model hyper params
MODEL_HYPER_PARAMS = {
    "model_path": "./",
    "inference_token_len": 48,
}


class Aurora(FoundationForecastingModelBase):
    """
    FITS adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(Aurora, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "Aurora"

    def _init_model(self):
        model_path = self.config.model_path
        model = AuroraForPrediction.from_pretrained(model_path)
        return model

    def _process(self, input, dec_input, input_mark, target_mark):
        n_vars = input.shape[-1]
        input = rearrange(input, "b l c -> (b c) l")
        output = self.model.generate(inputs=input, max_output_length=self.config.horizon,
                                     inference_token_len=self.config.inference_token_len, num_samples=100)
        output = rearrange(output, "(b c) s l -> s b l c", c=n_vars)
        output = output.mean(0)
        return {"output": output}
