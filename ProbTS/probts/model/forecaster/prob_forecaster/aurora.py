import torch
from einops import rearrange
from probts.model.forecaster import Forecaster

from probts.model.nn.prob.aurora.modeling_aurora import AuroraForPrediction


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class Aurora(Forecaster):
    def __init__(
            self,
            model_path,
            inference_token_len=48,
            **kwargs
    ):
        """
        Initialize the model with parameters.
        """
        super().__init__(**kwargs)
        # Initialize model parameters here
        self.inference_token_len = inference_token_len
        self.model_path = model_path

        self.model = AuroraForPrediction.from_pretrained(self.model_path)
        self.no_training = True

    def sample_from_distribution(self, input, num_samples):
        input = rearrange(input, 'b l c -> (b c) l')
        samples = self.model.generate(inputs=input, max_output_length=self.prediction_length, num_samples=num_samples, inference_token_len=self.inference_token_len)
        return rearrange(samples, '(b c) s l -> b s l c', c=self.input_size)

    def forecast(self, batch_data, num_samples=None):
        """
        Generate forecasts for the given batch data.

        Parameters:
        batch_data [dict]: Dictionary containing input data.
        num_samples [int, optional]: Number of samples per distribution during evaluation. Defaults to None.

        Returns:
        Tensor: Forecasted outputs.
        """
        # Perform the forward pass to get the outputs
        self.model.eval()
        input = batch_data.past_target_cdf[:, -self.context_length:, :]
        with torch.no_grad():
            if num_samples is not None:
                # If num_samples is specified, use it to sample from the distribution
                outputs = self.sample_from_distribution(input, num_samples)
            else:
                outputs = None
        return outputs  # [batch_size, num_samples, prediction_length, var_num]


