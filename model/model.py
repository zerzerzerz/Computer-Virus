from statistics import mode
import torch
from torch import nn

class Model(nn.Module):
    def __init__(
        self,
        input_dim = 1024,
        output_dim = 2,
        num_layers = 3
    ):
        super().__init__()
        model = []
        in_d = input_dim
        for out_d in torch.linspace(input_dim, output_dim, num_layers).to(dtype=torch.int32):
            model.append(nn.Linear(in_d, out_d.item()))
            in_d = out_d
            model.append(nn.LeakyReLU(0.2))
        model.pop(-1)
        # model.append(nn.Softmax(dim = -1))
        self.model = nn.Sequential(*model)
    
    def forward(self,img):
        """
        img.shape = B * D_in
        """
        return self.model(img)
            