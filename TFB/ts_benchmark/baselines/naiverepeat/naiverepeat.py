import torch.nn as nn
from ts_benchmark.baselines.foundation_forecasting_model_base import FoundationForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
}


class NaiveRepeatModel(nn.Module):
    def __init__(self, config):
        super(NaiveRepeatModel, self).__init__()
        self.config = config

    def forward(self, input):
        repeat_value = input[:, -1, :].unsqueeze(1)
        outputs = repeat_value.repeat(1, self.config.horizon, 1)
        return outputs


class NaiveRepeat(FoundationForecastingModelBase):
    """
    FITS adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(NaiveRepeat, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "NaiveRepeat"

    def _init_model(self):
        model = NaiveRepeatModel(self.config)
        return model

    def _process(self, input, dec_input, input_mark, target_mark):
        output = self.model(input)
        return {"output": output}
