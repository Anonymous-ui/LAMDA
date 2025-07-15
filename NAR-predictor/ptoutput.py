import torch.nn as nn
import torch

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.avgpool(x)

        return x


model = CustomModel()

model_path = 'C:\\Users\\UserName\\Desktop\\qwen\\1.pt'
torch.save(model.state_dict(), model_path)
print(f"Model saved to '{model_path}'")