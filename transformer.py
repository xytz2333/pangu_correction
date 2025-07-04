import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

matplotlib.rc("font",family='YouYuan')

# 读取数据
data_processed = pd.read_csv("2024年综合数据.csv")

# 数据预处理类
class WeatherDataset(Dataset):
    def __init__(self, data, hist_len=25, pred_len=70):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.scalers = {}
        self.real_values = []

        # 提取特征并标准化
        features = ['U_predict', 'V_predict', 'temp_predict',
                    'U_real', 'V_real', 'temp_real',
                    'U_diff', 'V_diff', 'temp_diff']

        # 单独标准化每个特征组
        for group in ['predict', 'real', 'diff']:
            cols = [c for c in features if group in c]
            self.scalers[group] = StandardScaler()
            data[cols] = self.scalers[group].fit_transform(data[cols])

        # 添加时间特征
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['hour_sin'] = np.sin(2 * np.pi * data['datetime'].dt.hour / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['datetime'].dt.hour / 24)
        data['doy_sin'] = np.sin(2 * np.pi * data['datetime'].dt.dayofyear / 365)
        data['doy_cos'] = np.cos(2 * np.pi * data['datetime'].dt.dayofyear / 365)

        # 构建样本
        self.X_enc, self.X_dec, self.y = [], [], []
        for i in range(len(data) - hist_len - pred_len):

            # 编码器输入：历史25小时数据（9个特征 + 4个时间特征）
            enc_feats = data[features + ['hour_sin', 'hour_cos', 'doy_sin', 'doy_cos']].values[i:i + hist_len]

            # 解码器输入：未来70小时预测数据（3个特征）
            dec_feats = data[['U_predict', 'V_predict', 'temp_predict']].values[i + hist_len:i + hist_len + pred_len]

            # 目标：未来70小时的差值
            target = data[['U_diff', 'V_diff', 'temp_diff']].values[i + hist_len:i + hist_len + pred_len]

            real_feats = data[['U_real', 'V_real', 'temp_real']].values[i + hist_len:i + hist_len + pred_len]

            self.real_values.append(real_feats)
            self.X_enc.append(enc_feats)
            self.X_dec.append(dec_feats)
            self.y.append(target)

    def __len__(self):
        return len(self.X_enc)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X_enc[idx]),
            torch.FloatTensor(self.X_dec[idx]),
            torch.FloatTensor(self.y[idx]),
            torch.FloatTensor(self.real_values[idx])
        )

# Transformer模型
class WeatherTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=5, num_decoder_layers=5):
        super().__init__()

        # 编码器部分
        self.enc_embed = nn.Linear(13, d_model)  # 9个特征 + 4个时间特征
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 解码器部分
        self.dec_embed = nn.Linear(3, d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model * 4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 输出层
        self.output_layer = nn.Linear(d_model, 3)

    def forward(self, enc_input, dec_input):

        # 编码器部分
        enc_embed = self.enc_embed(enc_input)  # [batch_size, 25, d_model]
        enc_embed = self.pos_encoder(enc_embed)  # 编码器位置编码
        enc_embed = enc_embed.permute(1, 0, 2)  # 调整为 [25, batch_size, d_model]
        memory = self.encoder(enc_embed)

        # 解码器部分
        dec_embed = self.dec_embed(dec_input)  # [batch_size, 70, d_model]
        dec_embed = self.pos_decoder(dec_embed)  # 解码器位置编码
        dec_embed = dec_embed.permute(1, 0, 2)  # 调整为 [70, batch_size, d_model]
        output = self.decoder(dec_embed, memory)

        # 输出
        diff_pred = self.output_layer(output).permute(1, 0, 2)  # [batch_size, 70, 3]
        return diff_pred

# 训练参数
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20

# 准备数据
dataset = WeatherDataset(data_processed)
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
test_indices = test_dataset.indices
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(1), :].unsqueeze(0)  # [1, seq_len, d_model]
        pe = pe.repeat(x.size(0), 1, 1)  # [batch_size, seq_len, d_model]
        return x + pe.to(x.device)

# 模型损失函数
class PhysicsInformedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha  # 观测拟合项权重
        self.beta = beta  # 背景场偏差项权重

    def forward(self, outputs, targets):
        """
        outputs: 模型预测的差值 [batch_size, pred_len, 3]
        targets: 真实差值 [batch_size, pred_len, 3]
        """

        # 1. 观测拟合项 (数据匹配损失)
        obs_loss = nn.MSELoss()(outputs, targets)

        # 2. 背景场偏差项 (修正量正则化)
        background_loss = torch.mean(outputs ** 2)

        # 总损失
        total_loss = (
                self.alpha * obs_loss +
                self.beta * background_loss
        )
        return total_loss

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WeatherTransformer().to(device)
criterion = PhysicsInformedLoss(
    alpha=1.0,
    beta=0.1,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for enc_input, dec_input, targets, _ in train_loader:
        enc_input, dec_input, targets = enc_input.to(device), dec_input.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(enc_input, dec_input)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for enc_input, dec_input, targets, _ in test_loader:
            enc_input, dec_input, targets = enc_input.to(device), dec_input.to(device), targets.to(device)
            outputs = model(enc_input, dec_input)
            test_loss += criterion(outputs, targets).item()

    print(
        f'Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss / len(train_loader):.4f} | Test Loss: {test_loss / len(test_loader):.4f}')

# 可视化函数
def plot_results(sample_idx=0):
    model.eval()
    with torch.no_grad():
        enc_input, dec_input, targets, real_targets = test_dataset[sample_idx]
        original_idx = test_indices[sample_idx]

        # 计算时间窗口起点
        start_time_idx = original_idx + 25
        if start_time_idx + 70 > len(data_processed):
            start_time_idx = len(data_processed) - 70
        time_range = pd.date_range(
            start=data_processed['datetime'].iloc[start_time_idx],
            periods=70,
            freq='H'
        )
        enc_input = enc_input.unsqueeze(0).to(device)
        dec_input = dec_input.unsqueeze(0).to(device)
        pred_diff = model(enc_input, dec_input).cpu().numpy()[0]

    # 逆标准化
    _, _, _, real_targets = test_dataset[sample_idx]
    pred_diff = dataset.scalers['diff'].inverse_transform(pred_diff)
    orig_predict = dataset.scalers['predict'].inverse_transform(dec_input[0].cpu().numpy())
    orig_real = dataset.scalers['real'].inverse_transform(real_targets.cpu().numpy())

    # 生成修正后的预测
    corrected = orig_predict + pred_diff

    # 绘制曲线图
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    variables = ['经度方向风速', '纬度方向风速','温度']
    units = [ '(m/s)', '(m/s)','(K)']

    for i in range(3):
        ax = axes[i]
        ax.plot(time_range, orig_predict[:, i], color='green',linewidth=3,label='盘古预报')
        ax.plot(time_range, orig_real[:, i], color='red',linewidth=3,label='气象站实测')
        ax.plot(time_range, corrected[:, i], color='blue',linewidth=3,label='模型修正')
        ax.set_title(f'{variables[i]}对比图{units[i]}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# 显示第一个测试样本的结果
plot_results(sample_idx=0)


