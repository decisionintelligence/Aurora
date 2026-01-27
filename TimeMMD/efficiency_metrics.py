import torch
from thop import profile
from thop.utils import clever_format
import inspect
from aurora.modeling_aurora import AuroraForPrediction

class MultimodalMACsCounter:
    """多模态模型MACs统计工具类"""

    @staticmethod
    def get_input_order(model):
        """获取模型forward方法的输入参数顺序"""
        sig = inspect.signature(model.forward)
        return [param for param in sig.parameters.keys() if param != 'self']

    @staticmethod
    def count(model, input_data):
        """
        统计多模态模型的MACs和参数量

        参数:
            model: 多模态模型实例
            input_data: 字典形式的输入数据，键为参数名，值为对应的输入张量
        """
        # 获取输入参数顺序
        input_order = MultimodalMACsCounter.get_input_order(model)

        # 按模型要求的顺序准备输入元组
        input_tuple = tuple(input_data[param] for param in input_order if param in input_data)

        # 计算MACs和参数量
        macs, params = profile(model, inputs=input_tuple, verbose=False)

        # 格式化输出
        macs, params = clever_format([macs, params], "%.3f")
        return macs, params


# ------------------------------
# 使用示例：文本-图像多模态模型
# ------------------------------
if __name__ == "__main__":

    input_data = {
        "input_ids": torch.randn(1, 96,device='cpu'),  # (batch, seq_len, hidden_dim)
        "text_input_ids": torch.randint(low=0, high=10, size=(1, 10), device='cpu', dtype=torch.long),  # (batch, channel, h, w)
        "text_attention_mask": torch.randint(low=0, high=2, size=(1, 10), device='cpu', dtype=torch.long),  # 可选输入
        "text_token_type_ids": torch.randint(low=0, high=2, size=(1, 10), device='cpu', dtype=torch.long),  # 可选输入
        "vision_ids": None
    }

    # 3. 统计MACs和参数量
    model = AuroraForPrediction.from_pretrained('/home/Aurora/checkpoints/Aurora_Multi_Modal_First_Version')
    model.to('cpu')
    macs, params = MultimodalMACsCounter.count(model, input_data)

    print(f"多模态模型统计结果:")
    print(f"MACs: {macs}")
    print(f"参数量: {params}")
