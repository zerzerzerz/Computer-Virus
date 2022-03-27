from tabnanny import check
from model.model import Model
from function.function import *
from easydict import EasyDict
import torch


config = EasyDict(load_json('config/config.json'))

checkpoint_path = None
model = Model()
run(model,config, mode='train', checkpoint_path=checkpoint_path)

# checkpoint_path = 'results/000014/checkpoints/9.pt'
# model = torch.load(checkpoint_path)
# run(model,config, mode='test', checkpoint_path=checkpoint_path)