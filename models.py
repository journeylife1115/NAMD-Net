"""Core neural-network modules used by NAMD-Net."""
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=24, dropout=0.5):
        super().__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class Chomp1d(nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n
    def forward(self, x):
        return x if self.n == 0 else x[:, :, :-self.n]

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp = Chomp1d(pad)
    def forward(self, x):
        return self.chomp(self.conv(x))

class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)
    def forward(self, x):
        z = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        z = self.dropout(z)
        return x + self.residual(z), self.skip(z)

class WaveNetModel(nn.Module):
    def __init__(self, input_size, channels=48, output_size=24, kernel_size=2,
                 dropout=0.15, dilations=(1,2,4,8,16,32), skip_channels=32):
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, channels, 1)
        self.blocks = nn.ModuleList(
            WaveNetBlock(channels, kernel_size, d, dropout) for d in dilations
        )
        self.skip_projection = nn.Conv1d(channels, skip_channels, 1)
        self.output_projection = nn.Sequential(
            nn.ReLU(), nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(skip_channels, output_size, 1)
        )

    def forward(self, x):
        # input: (batch, features, time)
        x = self.input_projection(x)
        skip_sum = None
        for block in self.blocks:
            x, skip = block(x)
            skip = self.skip_projection(skip)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        return self.output_projection(skip_sum)[:, :, -1]

relate_num = 0;#Changes Based on Permutation Entropy Results
def select_branch(group_id, input_size, output_size=24):
    if group_id <= relate_num:
        return WaveNetModel(input_size=input_size, output_size=output_size)
    return LSTMModel(input_size=input_size, output_size=output_size)
