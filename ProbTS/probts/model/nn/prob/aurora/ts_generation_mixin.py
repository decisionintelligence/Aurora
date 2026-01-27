import os
from typing import Any, Dict, List, Optional, Union, Callable

import torch
from transformers import BertTokenizer
from transformers import GenerationMixin, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerationConfig, GenerateOutput
from transformers.utils import ModelOutput


class TSGenerationMixin(GenerationMixin):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert_config'), local_files_only=True)

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            text_inputs=None,
            text_input_ids: Optional[torch.Tensor] = None,
            text_attention_mask: Optional[torch.Tensor] = None,
            text_token_type_ids: Optional[torch.Tensor] = None,
            vision_inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            revin: Optional[bool] = True,
            num_samples: Optional[int] = 1,
            max_output_length: Optional[int] = 96,
            inference_token_len: Optional[int] = None,
            max_text_token_length: Optional[int] = 125,
            **kwargs,
    ) -> Union[GenerateOutput, torch.Tensor]:
        if len(inputs.shape) != 2:
            raise ValueError('Input shape must be: [batch_size, seq_len]')
        if revin:
            means = inputs.mean(dim=-1, keepdim=True)
            stdev = inputs.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
            inputs = (inputs - means) / stdev
        if text_inputs is not None:
            tokenized_text = self._tokenize(text_inputs, max_length=max_text_token_length)
            text_input_ids = tokenized_text['input_ids'].squeeze(0)
            text_attention_mask = tokenized_text['attention_mask'].squeeze(0)
            text_token_type_ids = tokenized_text.get('token_type_ids', torch.zeros_like(text_input_ids)).squeeze(0)

        model_inputs = self.prepare_inputs_for_generation(
            inputs,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_token_type_ids=text_token_type_ids,
            vision_inputs=vision_inputs,
            generation_config=generation_config,
            max_output_length=max_output_length,
            inference_token_len=inference_token_len,
            **kwargs
        )

        outputs = self(**model_inputs, return_dict=True, revin=False, num_samples=num_samples)

        predictions = outputs.logits

        if revin:
            stdev = stdev.unsqueeze(1).repeat(1, num_samples, 1)
            means = means.unsqueeze(1).repeat(1, num_samples, 1)
            predictions = (predictions * stdev) + means

        return predictions

    def prepare_inputs_for_generation(
            self,
            inputs: torch.Tensor,
            text_input_ids: Optional[torch.Tensor] = None,
            text_attention_mask: Optional[torch.Tensor] = None,
            text_token_type_ids: Optional[torch.Tensor] = None,
            vision_inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            max_output_length: Optional[int] = None,
            inference_token_len: Optional[int] = None,
            **kwargs
    ):
        return {
            "input_ids": inputs,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_token_type_ids": text_token_type_ids,
            "vision_ids": vision_inputs,
            "max_output_length": max_output_length,
            "inference_token_len": inference_token_len,
            **kwargs
        }

    def _tokenize(self, texts, max_length):
        return self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            horizon_length: int = 1,
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        return model_kwargs
