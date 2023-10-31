import torch
import torch.nn as nn


x = torch.randn(2, 3)
print(x)
print(nn.Parameter(x))
