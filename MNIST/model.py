import torch.nn as nn

batch_size = 256

# torch.nn.Conv2d(
#     in_channels, # 입력 채널의 수
#     out_channels, # 출력 채널의 수 -> 즉 필터의 개수를 알 수 있음
#     kernel_size, # 커널의 크기 = 필터의 크기
#     stride=1, # 기본값들
#     padding=0, 
#     dilation=1, 
#     groups=1, 
#     bias=True, 
#     padding_mode='zeros'
# )

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out