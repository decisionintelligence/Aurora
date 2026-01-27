from aurora.modeling_aurora import AuroraForPrediction as AuroraForPredictionMultiModal
from aurora.configuration_aurora import AuroraConfig as AuroraConfigMultiModal
from aurora_uni_modal.modeling_aurora import AuroraForPrediction as AuroraForPredictionUniModal
from transformers import BertModel, ViTModel

multi_config = AuroraConfigMultiModal.from_json_file('/home/Aurora/aurora/config.json')
multi_model = AuroraForPredictionMultiModal(multi_config)

# inherit params from uni_modal aurora
uni_model = AuroraForPredictionUniModal.from_pretrained('/home/Aurora/logs/aurora_1_9_all/checkpoint-55530')
uni_model_params = uni_model.state_dict()
multi_model.load_state_dict(uni_model_params, strict=False)

# inherit params from bert
origin_bert = BertModel.from_pretrained('/home/Aurora/checkpoints/google-bert_bert-base-uncased')
multi_model.model.TextEncoder.model.load_state_dict(origin_bert.state_dict())

# inherit params from vit
origin_vit = ViTModel.from_pretrained('/home/Aurora/checkpoints/google_vit-base-patch16-224-in21k')
multi_model.model.VisionEncoder.model.load_state_dict(origin_vit.state_dict())

# Save
multi_model.save_pretrained('/home/Aurora/checkpoints/Aurora_Multi_Modal')