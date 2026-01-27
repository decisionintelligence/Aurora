import numpy as np
import torch
import torch.nn as nn

from .configuration_aurora import AuroraConfig
from .util_functions import sinusoidal_position_embedding, causal_attention_mask


class PrototypeRetriever(nn.Module):
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_prototypes = config.num_prototypes
        self.token_len = config.token_len

        # Define the learnable prototype parameter container.
        # Initialize an empty Parameter first, to be filled in _initialize_prototypes.
        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, self.token_len))

        # Initialize prototypes using the new logic
        self._initialize_prototypes()

        self.retriever = Retriever(config)

    def _initialize_prototypes(self, random_seed=42):
        """
        Initialize prototype parameters using diverse function generators.
        Adapted from the generate_prototypes logic to fit the class structure.
        """
        # Set random seed for reproducibility
        np.random.seed(random_seed)

        length = self.token_len
        # Create time series x, range from 0 to 10
        x = np.linspace(0, 10, length)

        prototypes_list = []

        # --- Define internal generation functions ---
        def generate_sin():
            """Generate sine function features"""
            freq = np.random.uniform(0.3, 2.0)
            amp = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, np.pi)
            return amp * np.sin(freq * x + phase)

        def generate_cos():
            """Generate cosine function features"""
            freq = np.random.uniform(0.3, 2.0)
            amp = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, np.pi)
            return amp * np.cos(freq * x + phase)

        def generate_log():
            """Generate logarithmic function features (trend)"""
            # Ensure x is positive, suitable for log function
            x_log = x + np.random.uniform(0.5, 2.0)
            slope = np.random.uniform(0.3, 1.5)
            offset = np.random.uniform(-2.0, 2.0)
            return slope * np.log(x_log) + offset

        def generate_exponential():
            """Generate exponential function features (trend)"""
            # Can be positive or negative, allowing growth or decay
            growth = np.random.uniform(-0.3, 0.3)
            amp = np.random.uniform(0.5, 2.0)
            return amp * np.exp(growth * x)

        def generate_linear():
            """Generate linear function features (trend)"""
            slope = np.random.uniform(-1.0, 1.0)
            intercept = np.random.uniform(-2.0, 2.0)
            return slope * x + intercept

        def generate_combination():
            """Generate combined features from multiple functions"""
            # Generate weights that sum to 1
            weights = np.random.dirichlet(np.ones(3))
            func1 = generate_sin()
            func2 = generate_linear()
            # Randomly select the third component
            func3 = generate_exponential() if np.random.random() > 0.5 else generate_log()
            return weights[0] * func1 + weights[1] * func2 + weights[2] * func3

        # Function types and their probability distributions
        functions = [
            (generate_sin, 0.2),
            (generate_cos, 0.2),
            (generate_log, 0.15),
            (generate_exponential, 0.15),
            (generate_linear, 0.1),
            (generate_combination, 0.2)
        ]

        # Extract functions and corresponding probabilities
        funcs, probs = zip(*functions)

        # --- Prototype generation loop ---
        for _ in range(self.num_prototypes):
            # Randomly select function type based on probability
            func = np.random.choice(funcs, p=probs)
            prototype = func()

            # Add some noise
            noise_level = np.random.uniform(0.05, 0.2)
            noise = np.random.normal(0, noise_level, length)
            prototype += noise

            prototypes_list.append(prototype)

        # Convert to Numpy array
        prototypes_np = np.array(prototypes_list)

        # --- Key step: Convert to Tensor and assign to Parameter ---
        # 1. Convert to Tensor
        # 2. Convert to float32 (numpy defaults to float64, PyTorch typically uses float32)
        # 3. Use .data.copy_ to fill nn.Parameter, maintaining the gradient tracking mechanism
        tensor_data = torch.from_numpy(prototypes_np).float()
        self.prototypes.data.copy_(tensor_data)

    def forward(self, x, output_token_len):
        """
        Args:
            x: Input representation with shape [B, k, d]
        Returns:
            synthetic_protos: [B, F, p] (Normalized)
        """
        # Calculate distribution [B, F, M]
        dist = self.retriever(x, output_token_len)

        # Weighted combination of prototypes [B, F, p]
        synthetic_protos = torch.matmul(dist, self.prototypes)

        # Normalize
        # Note: Since the new initialization logic generates values with larger ranges and noise,
        # Instance Normalization here is crucial for output stability.
        mean = synthetic_protos.mean(dim=-1, keepdim=True).detach()
        std = synthetic_protos.std(dim=-1, keepdim=True).detach() + 1e-5
        synthetic_protos = (synthetic_protos - mean) / std

        return synthetic_protos


class Retriever(nn.Module):
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.input_emb = nn.Sequential(nn.LayerNorm(config.hidden_size),
                                       nn.Linear(config.hidden_size, config.hidden_size))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout_rate,
                batch_first=True,
            ),
            norm=nn.LayerNorm(config.hidden_size),
            num_layers=config.num_retriever_enc_layers,
        )
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout_rate,
                batch_first=True,
            ),
            norm=nn.LayerNorm(config.hidden_size),
            num_layers=config.num_retriever_dec_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),  # Combine context and position information
            nn.LayerNorm(config.intermediate_size),
            nn.SiLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.intermediate_size, config.num_prototypes),  # Predict prototype distribution
            nn.Softmax(dim=-1)
        )

        self.hidden_size = config.hidden_size

    def forward(self, x, output_token_len):
        x_encoded = self.input_emb(x)
        enc_attn_mask = causal_attention_mask(x.shape[1]).to(x.device)
        enc_output = self.encoder(x_encoded, mask=enc_attn_mask.squeeze(0).squeeze(0))  # Shape: [B, k, d]

        enc_output = enc_output[:, -1:, :]

        dec = enc_output.repeat(1, output_token_len, 1)

        pos_embeds = sinusoidal_position_embedding(
            batch_size=dec.shape[0], num_heads=1,
            max_len=output_token_len, output_dim=self.hidden_size,
            device=dec.device).squeeze(1)

        embeds = dec + pos_embeds

        dec_attn_mask = causal_attention_mask(output_token_len).to(x.device)
        dec_output = self.decoder(embeds, mask=dec_attn_mask.squeeze(0).squeeze(0))

        dist = self.head(dec_output)  # Shape: [B, F, M]

        return dist
