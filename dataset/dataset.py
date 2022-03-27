import torch
from torch.utils.data import Dataset
import json
# from function.function import load_json

def load_json(path):
    with open(path,'r') as f:
        ans = json.load(f)
    return ans

class MyDataset(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        data = load_json(path)
        self.data = []
        for item in data:
            img = torch.Tensor(item[0])
            label = item[1]

            # benign
            if label == 0:
                label = 0
            # virus
            else:
                label = 1
            label = torch.IntTensor([label])

            self.data.append((img, label))
        self.length = len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.length
        

