import torch.nn as nn


class PrepBlock(nn.Module):
  # fixed channels for cifar
  # from the paper: 
  # We adopt batch normalization (BN) [16] right after each convolution and
  # before activation, following [16].
  def __init__(self):
    super().__init__()
    self.prep_block = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
  def forward(self, x):
    return self.prep_block(x)
  def init_weights(self):
    for layer in self.prep_block:
      if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
          layer.bias.data.zero_()

class ComputeBlock(nn.Module):
  def __init__(self, inchannels, outchannels, stride, downsample=None):
    super().__init__()
    self.convblock = nn.Sequential(
      nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, padding=1, stride=stride, bias=False),
      nn.BatchNorm2d(outchannels),
      nn.ReLU(),
      nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3, padding=1, stride=1, bias=False),
      nn.BatchNorm2d(outchannels)
    )
    self.downsample = downsample
    if not inchannels == outchannels:
      self.downsample = nn.Sequential(
        nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(outchannels)
      )
    self.relu = nn.ReLU()
  def forward(self, x):
    x_skip = x 
    x = self.convblock(x)
    if self.downsample:
      x_skip = self.downsample(x_skip)
    out = self.relu(x+x_skip)
    return out
  def init_weights(self):
    for layer in self.convblock:
      if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
          layer.bias.data.zero_()
    if self.downsample:
      for layer in self.downsample:
        if isinstance(layer, nn.Conv2d):
          nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
          if layer.bias is not None:
            layer.bias.data.zero_()
  
class ResNet18New(nn.Module):
  def __init__(self):
    super().__init__()
    self.resnet = nn.Sequential(
      PrepBlock(), # (B,64,8,8)
      ComputeBlock(64, 64, stride=1), # (B,64,8,8)
      ComputeBlock(64,128, stride=2), # (B,128,4,4)
      ComputeBlock(128,256, stride=2), # (B,256,2,2)
      ComputeBlock(256, 512, stride=2), # (B,512,1,1)
      nn.AdaptiveAvgPool2d((1,1)),
      nn.Flatten(start_dim=1), # (B,512)
      nn.Linear(512, 10) # (B, 10)
    )
  def forward(self, x):
    return self.resnet(x)
  def init_weights(self):
    for module in self.modules():
      if isinstance(module, ComputeBlock):
        module.init_weights()
      elif isinstance(module, PrepBlock):
        module.init_weights()
      elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
