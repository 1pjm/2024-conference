import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: 데이터 불러오기 및 전처리
class TurnTimeDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        self.sequence_length = sequence_length
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Excel 파일에서 데이터 불러오기
file_path = 'data\사이버로지텍_pnc-gate-data.xlsx'
df = pd.read_excel(file_path)

# NaN 값 확인 및 제거
if df['TURNTIME'].isnull().any():
    df = df.dropna(subset=['TURNTIME'])

# TURNTIME 열 추출 및 데이터 정규화
turntime_data = df['TURNTIME'].values
turntime_data = (turntime_data - np.mean(turntime_data)) / np.std(turntime_data)

# 데이터셋 및 데이터로더 생성
sequence_length = 10
batch_size = 64  # 배치 크기를 늘림
dataset = TurnTimeDataset(turntime_data, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 2: RNN 모델 정의
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):  # 숨겨진 크기 조정
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.RNN):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Step 3: 모델 학습
def train_model(model, dataloader, num_epochs=100, learning_rate=0.0001):  # 학습률을 낮춤
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(-1)  # 특성 차원 추가
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 모델 초기화 및 학습
model = RNNModel()
train_model(model, dataloader)

# Step 4: 모델 평가
def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(-1)
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(targets.tolist())
    return predictions, actuals

# 모델 평가
predictions, actuals = evaluate_model(model, dataloader)

# 예측값과 실제값을 원래 스케일로 변환
predictions = np.array(predictions) * np.std(turntime_data) + np.mean(turntime_data)
actuals = np.array(actuals) * np.std(turntime_data) + np.mean(turntime_data)

# 결과 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(actuals, label='Actual Turn Time')
plt.plot(predictions, label='Predicted Turn Time')
plt.legend()
plt.show()
