import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from transformers import ViTImageProcessor, ViTModel, BertModel, ViTConfig, BertConfig

from .configuration_aurora import AuroraConfig


class VisionEncoder(nn.Module):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_config')
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.processor = UnifiedImageProcessor(config)
        self.model = ViTModel(ViTConfig.from_json_file(os.path.join(self.config_path, 'config.json')))
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = self.model.config.hidden_size
        self.output_dim = config.hidden_size
        self.num_distill = config.num_distill

        self.projection = nn.Linear(self.hidden_size, self.output_dim)

        self.target_vision_tokens = nn.Parameter(torch.randn(self.num_distill, self.output_dim))

        # Cross-attention layer
        self.cross_vision = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout_rate,
                batch_first=True,
            ),
            norm=nn.LayerNorm(config.hidden_size),
            num_layers=config.num_vision_cross_layers,
        )

    def extract_vit_features(self, image_tensor):
        """
        Extract image features using ViT
        Args:
            image_tensor: Preprocessed image tensor with shape [batch_size, 3, H, W]
        Returns:
            cls_feature: [CLS] token feature with shape [batch_size, hidden_size]
            patch_features: Features of all patches with shape [batch_size, num_patches, hidden_size]
        """
        outputs = self.model(pixel_values=image_tensor)

        last_hidden_state = outputs.last_hidden_state

        cls_feature = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        patch_features = last_hidden_state[:, 1:, :]  # [batch_size, num_patches, hidden_size]

        return cls_feature, patch_features

    def forward(self, x, type='pseudo'):
        x = self.processor(x, type=type)
        _, patch_features = self.extract_vit_features(x)
        patch_features = self.projection(patch_features)
        target_vision_tokens = self.target_vision_tokens.unsqueeze(0).repeat(patch_features.shape[0], 1, 1)
        output_tokens = self.cross_vision(target_vision_tokens, patch_features)
        return output_tokens  # [batch_size, num_patches, hidden_size]


class UnifiedImageProcessor(nn.Module):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_config')
    def __init__(self, config: AuroraConfig):
        super().__init__()
        # Load ViT preprocessor to get pretrained normalization parameters and target size
        self.vit_processor = ViTImageProcessor.from_json_file(os.path.join(self.config_path, 'preprocessor_config.json'))
        self.target_size = self.vit_processor.size["height"]  # e.g., 224 (default ViT input size)

        # Define resizer for pseudo-images (matches real image target size)
        self.pseudo_resizer = Resize((self.target_size, self.target_size))

        self.token_len = config.token_len

    def process_real_image(self, images):
        """Process real images: automatic resizing, cropping, and normalization"""
        # Directly use ViTImageProcessor to ensure consistency with pretraining pipeline
        inputs = self.vit_processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]  # Shape: [batch_size, 3, H, W]

    def _period_search(self, x):
        xf = torch.fft.rfft(x, dim=-1)
        # find period by amplitudes
        frequency_list = abs(xf).mean(0)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, 1)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period

    def process_pseudo_image(self, x):
        """Process pseudo-images (converted from time series): ensure consistent normalization with real images"""

        # Segmentation
        input_length = x.shape[-1]
        period = list(self._period_search(x))[0]
        period = period if 0 < period < input_length else self.token_len
        if period > input_length:
            period = input_length

        padding_length = (period - (input_length %
                                            period)) % period
        x_pad = F.pad(x, (padding_length, 0))
        x_2d = einops.rearrange(x_pad, 'b (p f) -> b 1 f p', f=period)

        # 3. Render & Alignment
        x_resize = self.pseudo_resizer(x_2d)
        image_input = einops.repeat(x_resize, 'b 1 h w -> b c h w', c=3)
        return image_input

    def forward(self, x, type='pseudo'):
        if type == 'pseudo':
            return self.process_pseudo_image(x)
        else:
            return self.process_real_image(x)


class TextEncoder(nn.Module):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert_config')
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.model = BertModel(BertConfig.from_json_file(os.path.join(self.config_path, 'config.json')))
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = self.model.config.hidden_size
        self.output_dim = config.hidden_size
        self.num_distill = config.num_distill
        self.max_length = 125

        self.projection = nn.Linear(self.hidden_size, self.output_dim)

        # Define learnable target tokens (shape: [num_distill_tokens, hidden_size])
        self.target_text_tokens = nn.Parameter(torch.randn(self.num_distill, self.output_dim))

        self.cross_text = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout_rate,
                batch_first=True,
            ),
            norm=nn.LayerNorm(config.hidden_size),
            num_layers=config.num_text_cross_layers,
        )

    def extract_bert_features(self, input_dict):
        """Extract and clean BERT features with fixed output shape"""
        outputs = self.model(**input_dict)

        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        cls_feature = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        token_features = last_hidden_state

        # Create mask to exclude [CLS], [SEP], and padding tokens
        attention_mask = input_dict["attention_mask"]  # [batch_size, seq_len]
        batch_size, seq_len = attention_mask.shape
        valid_mask = torch.ones_like(attention_mask)
        valid_mask[:, 0] = 0  # Exclude [CLS]

        for i in range(batch_size):
            sep_pos = torch.where(attention_mask[i] == 1)[0][-1]
            valid_mask[i, sep_pos] = 0  # Exclude [SEP]

        # Apply mask and get valid tokens
        valid_token_mask = valid_mask.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        clean_token_features = token_features * valid_token_mask

        # Convert to fixed shape [batch_size, max_valid_tokens, hidden_size]
        fixed_features = torch.zeros(batch_size, self.max_length, self.hidden_size,
                                     device=clean_token_features.device)
        valid_counts = []

        for i in range(batch_size):
            # Get valid tokens (excluding zeros)
            valid_tokens = clean_token_features[i][clean_token_features[i].sum(dim=1) != 0]
            valid_count = valid_tokens.shape[0]
            valid_counts.append(valid_count)

            # Truncate if longer than max_length, else pad with zeros
            if valid_count > self.max_length:
                fixed_features[i] = valid_tokens[:self.max_length]
            else:
                fixed_features[i, :valid_count] = valid_tokens

        return cls_feature, token_features, fixed_features, valid_counts

    def forward(self, texts):
        """Return fixed-shape token features [batch_size, max_valid_tokens, hidden_size]"""
        _, _, fixed_features, _ = self.extract_bert_features(texts)
        fixed_features = self.projection(fixed_features)

        target_text_tokens = self.target_text_tokens.unsqueeze(0).repeat(fixed_features.shape[0], 1, 1)

        output_tokens = self.cross_text(target_text_tokens, fixed_features)
        return output_tokens


class ModalityConnector(nn.Module):
    def __init__(self, config: AuroraConfig):
        """
        Args:
            hidden_size: Feature dimension (must match text/vision feature dimensions)
            num_distill_tokens: Unified token count (constant N)
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        # Define learnable target tokens (shape: [num_distill_tokens, hidden_size])
        self.connect_text = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout_rate,
                batch_first=True,
            ),
            norm=nn.LayerNorm(config.hidden_size),
            num_layers=config.num_text_connect_layers,
        )

        self.connect_vision = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout_rate,
                batch_first=True,
            ),
            norm=nn.LayerNorm(config.hidden_size),
            num_layers=config.num_vision_connect_layers,
        )

    def forward(self, x, text_features, vision_features):
        """
        Distill text and vision tokens to the same count N
        Args:
            x: Time Series with shape [batch_size, n, hidden_size] (n is time series token count)
            text_features: Text features with shape [batch_size, T, hidden_size] (T is text token count)
            vision_features: Vision features with shape [batch_size, V, hidden_size] (V is vision token count)
        Returns:
            text_distilled: Distilled text tokens with shape [batch_size, N, hidden_size]
            vision_distilled: Distilled vision tokens with shape [batch_size, N, hidden_size]
        """
        if text_features is not None:
            from_text = self.connect_text(
                x,
                text_features
            )
        else:
            from_text = None

        from_vision = self.connect_vision(
            x,
            vision_features
        )

        return from_text, from_vision
