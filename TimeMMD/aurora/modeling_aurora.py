import random
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast

from .configuration_aurora import AuroraConfig
from .flow_loss import FlowLoss
from .modality_connector import ModalityConnector, VisionEncoder, TextEncoder
from .prototype_retriever import PrototypeRetriever
from .ts_generation_mixin import TSGenerationMixin
from .util_functions import resample, Transpose, causal_attention_mask, RoPE_decoder


class AuroraPatchEmbedding(nn.Module):
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.proj_layer = nn.Linear(config.token_len, config.hidden_size, bias=False)
        self.token_len = config.token_len
        self.threshold_ratio = config.threshold_ratio
        self.mask_ratio = config.mask_ratio

    def _freq_masking(self, x):
        x_fft = torch.fft.rfft(x, dim=-1)
        x_ifft_list = []
        for ratio in self.threshold_ratio:
            temp = x_fft.clone()
            truncation = int(temp.shape[-1] * ratio)
            if random.random() > self.mask_ratio:
                temp[:, :truncation] = 0
            else:
                temp[:, truncation:] = 0

            x_ifft = torch.fft.irfft(temp, dim=-1)
            x_ifft_list.append(x_ifft)
        x_ifft = torch.stack(x_ifft_list, dim=0)
        return rearrange(x_ifft, 's b l -> (s b) l')

    def _predict(self, x, inference_token_len=48):
        input_length = x.shape[-1]
        padding_length = (inference_token_len - (input_length %
                                                 inference_token_len)) % inference_token_len
        x = F.pad(x, (padding_length, 0))
        x = x.unfold(dimension=-1, size=inference_token_len,
                     step=inference_token_len)

        resampled_weight = resample(old=self.proj_layer.weight.data, new_patch_len=inference_token_len)

        output = F.linear(x, resampled_weight)

        return output, None

    def forward(self, x, inference_token_len=48):
        if not self.training:
            return self._predict(x, inference_token_len)

        input_length = x.shape[-1]
        padding_length = (self.token_len - (input_length %
                                            self.token_len)) % self.token_len
        x = F.pad(x, (padding_length, 0))

        x_masked = self._freq_masking(x)

        x_origin = x.unfold(dimension=-1, size=self.token_len,
                            step=self.token_len)
        output_origin = self.proj_layer(x_origin)

        x_masked = x_masked.unfold(dimension=-1, size=self.token_len,
                                   step=self.token_len)
        output_masked = self.proj_layer(x_masked)

        return output_origin, output_masked


class AuroraAttention(nn.Module):
    def __init__(self, config: AuroraConfig, layer_idx: Optional[int] = None, rope: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.dropout_rate
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rope = rope

    def _scaled_dot_product_attention(self, Q, K, V, bias=None, attn_mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(Q.size(-1))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
            else:
                attn_scores = attn_scores + attn_mask

        if bias is not None:
            if attn_scores.shape[0] > bias.shape[0]:
                bias = bias.repeat(attn_scores.shape[0] // bias.shape[0], 1, 1, 1)
            attn_scores += bias

        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.attention_dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = torch.matmul(attn_weights, V)

        return attn_output, attn_scores

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_embedding: torch.Tensor = None,
            value_embedding: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            bias: torch.Tensor = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        if key_embedding is None:
            key_embedding = hidden_states
        if value_embedding is None:
            value_embedding = hidden_states

        _, k_len, _ = key_embedding.size()
        _, v_len, _ = value_embedding.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(key_embedding)
        value_states = self.v_proj(value_embedding)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope:
            query_states, key_states = RoPE_decoder(query_states, key_states)

        attn_output, attn_scores = self._scaled_dot_product_attention(
            Q=query_states, K=key_states, V=value_states, bias=bias,
            attn_mask=attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_scores = None

        return attn_output, attn_scores


class AuroraFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(hidden_size, intermediate_size),
                                 ACT2FN[hidden_act],
                                 nn.Linear(intermediate_size, hidden_size))

    def forward(self, hidden_state):
        return self.ffn(hidden_state)


class AuroraDecoderLayer(nn.Module):
    def __init__(self, config: AuroraConfig, layer_idx: int):
        super().__init__()
        self.self_attn = AuroraAttention(config, layer_idx, rope=False)
        self.cross_attn = AuroraAttention(config, layer_idx, rope=True)

        self.ffn_layer = AuroraFFN(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )
        if config.norm_mode == 'batch':
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.hidden_size), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.hidden_size), Transpose(1, 2))
            self.norm3 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.hidden_size), Transpose(1, 2))
        else:
            self.norm1 = torch.nn.LayerNorm(config.hidden_size)
            self.norm2 = torch.nn.LayerNorm(config.hidden_size)
            self.norm3 = torch.nn.LayerNorm(config.hidden_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cross_states: torch.Tensor,
            output_attentions: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        residual = hidden_states

        num_token = hidden_states.shape[1]
        attention_mask = causal_attention_mask(num_token).to(hidden_states.device)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        x_attn = residual + self.norm1(hidden_states)

        x_cross, cross_attn_weights = self.cross_attn(hidden_states=x_attn, key_embedding=cross_states,
                                                      value_embedding=cross_states)
        x_cross = self.norm2(x_cross) + x_attn

        # Fully Connected
        output_states = self.ffn_layer(x_cross)
        output_states = self.norm3(output_states) + x_cross

        if not output_attentions:
            self_attn_weights = None
            cross_attn_weights = None

        return output_states, self_attn_weights, cross_attn_weights


class AuroraEncoderLayer(nn.Module):
    def __init__(self, config: AuroraConfig, layer_idx: int):
        super().__init__()
        self.self_attn = AuroraAttention(config, layer_idx, rope=False)
        self.ffn_layer = AuroraFFN(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )

        if config.norm_mode == 'batch':
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.hidden_size), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.hidden_size), Transpose(1, 2))
        else:
            self.norm1 = torch.nn.LayerNorm(config.hidden_size)
            self.norm2 = torch.nn.LayerNorm(config.hidden_size)

        self.dropout_1 = nn.Dropout(config.dropout_rate)
        self.dropout_2 = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: Optional[bool] = False,
            bias: torch.Tensor = None,
            **kwargs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        residual = hidden_states
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            bias=bias
        )
        x_attn = self.norm1(residual + self.dropout_1(hidden_states))

        # Fully Connected
        output_states = self.ffn_layer(x_attn)
        output_states = self.norm2(self.dropout_2(output_states) + x_attn)

        if not output_attentions:
            self_attn_weights = None

        return output_states, self_attn_weights


class AuroraPredictHead(nn.Module):
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.output_proj = nn.Linear(config.hidden_size, config.token_len, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def _predict(self, hidden_states: torch.Tensor, inference_token_len=48):
        resampled_weight = resample(old=self.output_proj.weight.data.T, new_patch_len=inference_token_len).T
        output = F.linear(hidden_states, resampled_weight)
        return output

    def forward(
            self,
            hidden_states: torch.Tensor,
            inference_token_len: int = 48,
            **kwargs
    ) -> torch.FloatTensor:
        if not self.training:
            return self._predict(hidden_states, inference_token_len)

        return self.output_proj(self.dropout(hidden_states))


class AuroraPreTrainedModel(PreTrainedModel):
    config_class = AuroraConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AuroraEncoderLayer", "AuroraDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = False


class AuroraModel(nn.Module):
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.embed_layer = AuroraPatchEmbedding(config)
        self.enc_layers = nn.ModuleList(
            [AuroraEncoderLayer(config, layer_idx)
             for layer_idx in range(config.num_enc_layers)]
        )
        self.dec_layers = nn.ModuleList(
            [AuroraDecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_dec_layers)]
        )
        self.mask_num = len(config.threshold_ratio)
        self.gradient_checkpointing = False

        self.VisionEncoder = VisionEncoder(config)
        self.TextEncoder = TextEncoder(config)
        self.ModalityConnector = ModalityConnector(config)

        self.VisionGuider = AuroraAttention(config)
        self.TextGuider = AuroraAttention(config)

        self.W = nn.Parameter(torch.eye(config.num_distill))
        self.fuse = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
            self,
            input_ids: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            text_input_ids: Optional[torch.FloatTensor] = None,
            text_attention_mask: Optional[torch.FloatTensor] = None,
            text_token_type_ids: Optional[torch.FloatTensor] = None,
            vision_ids: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            predict_token_num: Optional[int] = None,
            inference_token_len: Optional[int] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        # input_ids is the input of time series, its shape is [batch_size, seq_len]
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")
        if inference_token_len is None:
            inference_token_len = self.config.token_len

        masked_embeds = None
        if inputs_embeds is None:
            inputs_embeds, masked_embeds = self.embed_layer(input_ids, inference_token_len)

        if masked_embeds is None:
            x_enc = inputs_embeds
        else:
            x_enc = torch.concat([inputs_embeds, masked_embeds], dim=0)

        if vision_ids is not None:
            vision_features = self.VisionEncoder(vision_ids, type='real')
        else:
            vision_features = self.VisionEncoder(input_ids, type='pseudo')

        _, attn_vision = self.VisionGuider(
            inputs_embeds,
            vision_features,
            vision_features,
            output_attentions=True
        )

        if text_input_ids is not None:
            text_features = self.TextEncoder({'input_ids': text_input_ids, 'attention_mask': text_attention_mask,
                                              'token_type_ids': text_token_type_ids})
            _, attn_text = self.TextGuider(
                inputs_embeds,
                text_features,
                text_features,
                output_attentions=True
            )
        else:
            text_features = None
            attn_text = None

        if attn_text is not None:
            guided_bias = torch.einsum("bhik,kl,bhjl->bhij", attn_vision, self.W, attn_text)
        else:
            guided_bias = None

        # encoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for encoder_layer in self.enc_layers:
            if output_hidden_states:
                all_hidden_states += (x_enc,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    x_enc,
                    output_attentions,
                    guided_bias
                )
            else:
                layer_outputs = encoder_layer(
                    x_enc,
                    output_attentions=output_attentions,
                    bias=guided_bias
                )

            x_enc = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if x_enc.shape[0] > batch_size:
            x_enc, x_rec = torch.split(x_enc, [batch_size, x_enc.shape[0] - batch_size], dim=0)
            x_rec = rearrange(x_rec, '(s b) n d -> s b n d', s=self.mask_num)
            x_rec = x_rec.mean(0)
        else:
            x_rec = None

        decay_weights = 0.5 ** torch.arange(predict_token_num)
        decay_weights = decay_weights.unsqueeze(0).unsqueeze(-1).to(x_enc.device)

        from_text, from_vision = self.ModalityConnector(x_enc, text_features, vision_features)
        if from_text is not None:
            x_enc = x_enc + self.fuse(from_vision + from_text)
        else:
            x_enc = x_enc + self.fuse(from_vision)

        last_token = x_enc[:, -1:, :]
        x_dec = decay_weights * last_token.repeat(1, predict_token_num, 1)

        # decoder layers
        for decoder_layer in self.dec_layers:
            if output_hidden_states:
                all_hidden_states += (x_dec,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    x_dec,
                    x_enc,
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    x_dec,
                    x_enc,
                    output_attentions=output_attentions
                )

            x_dec = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (x_dec,)

        if not return_dict:
            return tuple(
                v
                for v in [x_dec, all_hidden_states, all_self_attns]
                if v is not None
            )

        output_states = (x_rec, x_dec, from_text, from_vision)

        return MoeModelOutputWithPast(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class AuroraForPrediction(AuroraPreTrainedModel, TSGenerationMixin):
    def __init__(self, config: AuroraConfig):
        super().__init__(config)
        self.config = config
        self.model = AuroraModel(config)
        self.point_loss = torch.nn.MSELoss(reduction='none')
        self.flow_match = FlowLoss(config.token_len, config.hidden_size, config.flow_loss_depth, config.hidden_size,
                                   config.num_sampling_steps)
        self.linear_head = AuroraPredictHead(config)

        self.retriever = PrototypeRetriever(config)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.FloatTensor = None,
            text_input_ids: torch.FloatTensor = None,
            text_attention_mask: torch.FloatTensor = None,
            text_token_type_ids: torch.FloatTensor = None,
            vision_ids: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            loss_masks: Optional[torch.FloatTensor] = None,
            mask_y: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            max_output_length: Optional[int] = None,
            revin: Optional[bool] = True,
            num_samples: Optional[int] = 1,
            inference_token_len: Optional[int] = 48,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if max_output_length is None:
                max_output_length = labels.shape[1]
            predict_token_num = math.ceil(max_output_length / self.config.token_len)
        else:
            predict_token_num = math.ceil(max_output_length / inference_token_len)

        if revin:
            means = input_ids.mean(1, keepdim=True).detach()
            stdev = input_ids.std(dim=1, keepdim=True, unbiased=False).detach() + 1e-5
            input_ids = (input_ids - means) / stdev

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_token_type_ids=text_token_type_ids,
            vision_ids=vision_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            predict_token_num=predict_token_num,
            inference_token_len=inference_token_len
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        x_rec, x_dec, from_text, from_vision = hidden_states

        if from_text is not None:
            generated_prototypes = self.retriever(from_text + from_vision, predict_token_num)
        else:
            generated_prototypes = self.retriever(from_vision, predict_token_num)

        loss = None
        predictions = None
        eps = 1e2
        mask = None
        if labels is not None:
            if revin:
                origin_labels = labels
                labels = (labels - means) / stdev

            origin_length = labels.shape[-1]
            target_length = predict_token_num * self.config.token_len
            if origin_length < target_length:
                pad_length = target_length - origin_length
                labels = F.pad(labels, (0, pad_length))
                mask = torch.tensor([1] * origin_length + [0] * pad_length, device=labels.device)
                mask = mask.unsqueeze(0)

            reco = rearrange(self.linear_head(x_rec), 'b n p -> b (n p)')
            fore = rearrange(self.linear_head(x_dec), 'b n p -> b (n p)')
            if revin:
                fore = fore * stdev + means

            reco_loss = self.point_loss(reco[:, :input_ids.shape[-1]], input_ids)
            fore_loss = self.point_loss(fore[:, :origin_length], origin_labels)
            reco_loss = reco_loss[reco_loss < eps]
            fore_loss = fore_loss[fore_loss < eps]
            point_loss = reco_loss.mean() + fore_loss.mean()

            shift_labels = labels.unfold(
                dimension=-1, size=self.config.token_len, step=self.config.token_len)
            bsz, L, _ = shift_labels.shape
            shift_labels = shift_labels.reshape(
                bsz * L, -1).repeat(self.config.diffusion_batch_mul, 1)
            x_dec = x_dec.reshape(
                bsz * L, -1).repeat(self.config.diffusion_batch_mul, 1)
            protos = generated_prototypes.reshape(bsz * L, -1).repeat(self.config.diffusion_batch_mul, 1)
            flow_loss = self.flow_match(target=shift_labels, z=x_dec.detach(), prototype=protos, eps=eps, mask=mask)
            loss = point_loss + flow_loss

        else:
            predictions = self.flow_match.sample(z=rearrange(x_dec, 'b n d -> (b n) d'),
                                                 prototype=rearrange(generated_prototypes, 'b n p -> (b n) p'),
                                                 num_samples=num_samples,
                                                 inference_token_len=inference_token_len)
            predictions = rearrange(predictions, '(b n) s p -> b s (n p)', n=predict_token_num)[:, :,
            :max_output_length]

            if revin:
                stdev = stdev.unsqueeze(1).repeat(1, num_samples, 1)
                means = means.unsqueeze(1).repeat(1, num_samples, 1)
                predictions = (predictions * stdev) + means

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=predictions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
