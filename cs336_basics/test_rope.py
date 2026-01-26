import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # 计算那行“玄学”频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        
        # 打印出来
        print("\n" + "="*30)
        print("🎉 恭喜！你成功运行了代码！")
        print(f"当 d_k 为 {d_k} 时，生成的频率表是：\n{inv_freq}")
        print("="*30 + "\n")

    def forward(self, x, token_positions):
        return x

# 实例化（让它真正跑起来）
if __name__ == "__main__":
    # 我们用 theta=10000, d_k=8 测试
    test_obj = RotaryPositionalEmbedding(theta=10000.0, d_k=8, max_seq_len=128)