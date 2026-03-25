import torch
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
class FourierAttention(nn.Module):
    def __init__(self, hidden_dim=768, attn_heads=8, dropout=0.1):
        super().__init__()
        # 频域自注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 频带能量聚合
        self.energy_proj = nn.Linear(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        输入: 
            x: [batch, seq_len, hidden_dim] (BERT输出的时域特征)
        输出:
            [batch, hidden_dim] (频域全局特征)
        """
        # 1. FFT变换到频域
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')  # [batch, seq_len//2+1, hidden_dim]
        mag, phase = x_freq.abs(), x_freq.angle()

        #plt.figure(figsize=(10, 4))
        #plt.plot(mag[0].cpu().detach().numpy())
        
        # 2. 计算频域能量（作为注意力权重）
        #energy = mag.pow(2).sum(dim=-1, keepdim=True)  # [batch, seq_len//2+1, 1]
        #attn_weights = F.softmax(self.energy_proj(energy), dim=1)  # [batch, seq_len//2+1, hidden_dim]
        
        # 3. 频域自注意力（加权聚合关键频率）
        attn_output, _ = self.attn(
            query=mag,  # 用能量加权作为query
            key=mag,
            value=mag
        )  # [batch, seq_len//2+1, hidden_dim]
        
        # 4. 全局池化（取均值）-> [batch, hidden_dim]
        global_feat = attn_output.mean(dim=1)
        return self.norm(global_feat)