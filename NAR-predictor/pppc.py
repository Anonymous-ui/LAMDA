import torch
import torch.onnx
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=14, kernel_size=5, stride=1, padding=2, dilation=1, groups=1),
            nn.Conv2d(in_channels=14, out_channels=48, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1, stride=2, padding=0, dilation=1, groups=48),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=28, kernel_size=5, stride=1, padding=2, dilation=1, groups=1),
            nn.Conv2d(in_channels=28, out_channels=144, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=144, out_channels=72, kernel_size=1, stride=1, padding=0, dilation=1, groups=72),
            nn.ReLU(),
            nn.Conv2d(in_channels=72, out_channels=28, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.Conv2d(in_channels=28, out_channels=144, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=144, out_channels=72, kernel_size=3, stride=1, padding=1, dilation=1, groups=72),
            nn.ReLU(),
            nn.Conv2d(in_channels=72, out_channels=28, kernel_size=5, stride=1, padding=2, dilation=1, groups=1),
            nn.Conv2d(in_channels=28, out_channels=144, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=144, out_channels=72, kernel_size=1, stride=2, padding=0, dilation=1, groups=72),
            nn.ReLU(),
            nn.Conv2d(in_channels=72, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=120, out_channels=120, kernel_size=1, stride=1, padding=0, dilation=1, groups=120),
            nn.ReLU(),
            nn.Conv2d(in_channels=120, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=120, out_channels=120, kernel_size=1, stride=1, padding=0, dilation=1, groups=120),
            nn.ReLU(),
            nn.Conv2d(in_channels=120, out_channels=16, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.Conv2d(in_channels=16, out_channels=480, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=480, out_channels=240, kernel_size=3, stride=2, padding=1, dilation=1, groups=240),
            nn.ReLU(),
            nn.Conv2d(in_channels=240, out_channels=104, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.Conv2d(in_channels=104, out_channels=480, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=480, out_channels=960, kernel_size=3, stride=1, padding=1, dilation=1, groups=480),
            nn.ReLU(),
            nn.Conv2d(in_channels=960, out_channels=104, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.Conv2d(in_channels=104, out_channels=480, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=480, out_channels=960, kernel_size=1, stride=1, padding=0, dilation=1, groups=480),
            nn.ReLU(),
            nn.Conv2d(in_channels=960, out_channels=104, kernel_size=5, stride=1, padding=2, dilation=1, groups=1),
            nn.Conv2d(in_channels=104, out_channels=480, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=480, out_channels=960, kernel_size=1, stride=1, padding=0, dilation=1, groups=480),
            nn.ReLU(),
            nn.Conv2d(in_channels=960, out_channels=113, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.Conv2d(in_channels=113, out_channels=576, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=576, out_channels=576, kernel_size=3, stride=1, padding=1, dilation=1, groups=576),
            nn.ReLU(),
            nn.Conv2d(in_channels=576, out_channels=113, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.Conv2d(in_channels=113, out_channels=576, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=576, out_channels=1152, kernel_size=3, stride=2, padding=1, dilation=1, groups=576),
            nn.ReLU(),
            nn.Conv2d(in_channels=1152, out_channels=266, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.Conv2d(in_channels=266, out_channels=2304, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2304, out_channels=2304, kernel_size=3, stride=1, padding=1, dilation=1, groups=1152),
            nn.ReLU(),
            nn.Conv2d(in_channels=2304, out_channels=266, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.Conv2d(in_channels=266, out_channels=1152, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1152, out_channels=2304, kernel_size=1, stride=1, padding=0, dilation=1, groups=1152),
            nn.ReLU(),
            nn.Conv2d(in_channels=2304, out_channels=266, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.Conv2d(in_channels=266, out_channels=1152, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1152, out_channels=2304, kernel_size=1, stride=1, padding=0, dilation=1, groups=1152),
            nn.ReLU(),
            nn.Conv2d(in_channels=2304, out_channels=266, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.Conv2d(in_channels=266, out_channels=1152, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1152, out_channels=266, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.Conv2d(in_channels=266, out_channels=1152, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1152, out_channels=1152, kernel_size=1, stride=1, padding=0, dilation=1, groups=1152),
            nn.ReLU(),
            nn.Conv2d(in_channels=1152, out_channels=268, kernel_size=5, stride=1, padding=2, dilation=1, groups=1),
            nn.Conv2d(in_channels=268, out_channels=1986, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1, count_include_pad=False, padding=0)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(17874, 2653)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2653, 5156)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(5156, 1986)

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = CustomModel()

# Set the model to evaluation mode
model.eval()
torch.save(model.state_dict(), 'D:\\NAR\\NAR-Former-V2-main\\NAR-Former-V2-main\\dataset3\\unseen_structure\\onnx\\myonnx\\6.5129.pt')

print(f"Model saved to ")
model.load_state_dict(torch.load('D:\\NAR\\NAR-Former-V2-main\\NAR-Former-V2-main\\dataset3\\unseen_structure\\onnx\\myonnx\\6.5129.pt'))
# Input to the model

dummy_input = torch.randn(1, 3,224, 224)  # Specify the input shape (batch_size, channels, height, width)

# Export the model to ONNX
onnx_path = 'D:\\NAR\\NAR-Former-V2-main\\NAR-Former-V2-main\\dataset3\\unseen_structure\\onnx\\myonnx\\6.5129.onnx'
torch.onnx.export(model, dummy_input, onnx_path, export_params=True, verbose=True)

print(f"Model saved to '{onnx_path}'")
