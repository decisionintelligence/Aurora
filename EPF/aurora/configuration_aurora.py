from transformers import PretrainedConfig


class AuroraConfig(PretrainedConfig):
    model_type = "aurora"

    def __init__(
            self,
            token_len: int = 48,
            hidden_size: int = 512,
            intermediate_size: int = 1024,
            num_enc_layers: int = 12,
            num_dec_layers: int = 12,
            num_attention_heads: int = 8,
            hidden_act: str = "silu",
            rope_theta: int = 10000,
            dropout_rate: float = 0.2,
            max_position_embeddings: int = 10000,
            num_sampling_steps: int = 50,
            flow_loss_depth: int = 3,
            diffusion_batch_mul: int = 4,
            threshold_ratio: list[float] = [0.2, 0.3, 0.4, 0.5],
            mask_ratio: float = 0.5,
            norm_mode: str = 'batch',
            num_prototypes: int = 1024,
            num_retriever_enc_layers: int = 1,
            num_retriever_dec_layers: int = 1,
            num_text_cross_layers: int = 1,
            num_vision_cross_layers: int = 1,
            num_text_connect_layers: int = 1,
            num_vision_connect_layers: int = 1,
            num_distill: int = 10,
            **kwargs,
    ):
        self.token_len = token_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta
        self.dropout_rate = dropout_rate
        self.max_position_embeddings = max_position_embeddings
        self.num_sampling_steps = num_sampling_steps
        self.flow_loss_depth = flow_loss_depth
        self.diffusion_batch_mul = diffusion_batch_mul
        self.threshold_ratio = threshold_ratio
        self.mask_ratio = mask_ratio
        self.norm_mode = norm_mode
        self.num_prototypes = num_prototypes
        self.num_retriever_enc_layers = num_retriever_enc_layers
        self.num_retriever_dec_layers = num_retriever_dec_layers
        self.num_text_cross_layers = num_text_cross_layers
        self.num_vision_cross_layers = num_vision_cross_layers
        self.num_text_connect_layers = num_text_connect_layers
        self.num_vision_connect_layers = num_vision_connect_layers
        self.num_distill = num_distill

        super().__init__(
            **kwargs,
        )
