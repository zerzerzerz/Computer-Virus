from email.mime import base
import json
import torch
import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.dataset import MyDataset
from tqdm import tqdm
from os import makedirs
from os.path import join
import pickle as pkl


def fetch_update_index():
    with open('index/index.txt', 'r+') as f:
        index = int(f.read())
        f.seek(0)
        f.truncate()
        f.write(str(index+1))
        index = "%06d" % index
        return index

def load_json(path):
    with open(path,'r') as f:
        ans = json.load(f)
    return ans

def save_json(obj, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f)

def load_pkl(pkl_path):
    with open(pkl_path,'rb') as f:
        m = pkl.load(f)
        return m

def save_pkl(obj,pkl_path):
    '''obj, path'''
    with open(pkl_path,'wb') as f:
        pkl.dump(obj,f)    

def normalize(tensor):
    """
    Normalize to [-1, 1]
    """
    return ((tensor) / 255 - 0.5) * 2

def get_loss(input, target):
    """
    input.shape = B * C, where C is the number of class
    target.shape = B, where 0 <= target value < C
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    return loss_fn(input, target.to(dtype=torch.int64))

def get_datetime():
    time1 = datetime.datetime.now()
    time2 = datetime.datetime.strftime(time1,'%Y-%m-%d-%H-%M-%S')
    return time2

def run(
    model,
    config,
    mode = "train",
    checkpoint_path = None
):
    # config = EasyDict(config)

    device = torch.device(config.device)
    config.checkpoint_path = checkpoint_path



    train_dataset = MyDataset(config.train_file_json_path,data_dim=min(config.data_dim, config.MAX_DATA_DIM))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_dataset = MyDataset(config.test_file_json_path, data_dim=min(config.data_dim, config.MAX_DATA_DIM))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)

    index = fetch_update_index()
    base_dir = join('results',index)
    makedirs(base_dir)
    save_json(dict(config), join(base_dir, 'config.json'))
    checkpoints_dir = join(base_dir,'checkpoints')
    makedirs(checkpoints_dir)
    train_log = join(base_dir,'train.txt')
    if mode == "train":
        with open(train_log,'w') as f:
            f.write(get_datetime() + '\n')
            f.write(str(model))
            f.write('\n')
    test_log = join(base_dir,'test.txt')
    with open(test_log,'w') as f:
        f.write(get_datetime() + '\n')
        f.write(str(model))
        f.write('\n')
    if  mode == "train":
        pass
    else:
        config.num_epoch = 1
        
    for epoch in tqdm(range(config.num_epoch)):
        if mode == "train":
            with torch.set_grad_enabled(True):
                model.train()
                loss_train = 0.0
                acc_train = 0.0
                count = 0
                for item in tqdm(train_dataloader):
                    img = normalize(item[0].to(device))
                    label = item[1].to(device).squeeze()
                    label_pred_distribution = model(img)

                    label_pred = label_pred_distribution.argmax(dim=1,keepdim=False).squeeze().to(dtype=torch.int32)
                acc_train += (label == label_pred).sum().item()
                count += label.shape[0]

                l = get_loss(label_pred_distribution, label)
                loss_train += l.sum().item()
                l = l.mean()
                l.backward()
                optimizer.step()
                optimizer.zero_grad()

            acc_train /= count
            loss_train /= count
            with open(train_log,'a') as f:
                f.write("Epoch: %d/%d, acc = %.4f, loss = %.4f\n" % (epoch, config.num_epoch, acc_train, loss_train))
            if((epoch + 1) % config.gap_epoch_save_checkpoint == 0):
                model.cpu()
                torch.save(model,join(checkpoints_dir,'%d.pt'%epoch))
                model.to(device=device)
        
        with torch.set_grad_enabled(False):
            model.eval()
            loss_test = 0.0
            acc_test = 0.0
            count = 0
            for item in tqdm(test_dataloader):
                # img = normalize(item[0][...,0:min(config.data_dim, 1024)].to(device))
                img = normalize(item[0].to(device))
                label = item[1].to(device).squeeze()
                label_pred_distribution = model(img)

                label_pred = label_pred_distribution.argmax(dim=1,keepdim=False).squeeze().to(dtype=torch.int32)
                acc_test += (label == label_pred).sum().item()
                count += label.shape[0]

                l = get_loss(label_pred_distribution, label)
                loss_test += l.sum().item()
                # l = l.mean()
                # l.backward()
                # optimizer.step()
                # optimizer.zero_grad()

            acc_test /= count
            loss_test /= count
            with open(test_log,'a') as f:
                f.write("Epoch: %d/%d, acc = %.4f, loss = %.4f\n" % (epoch, config.num_epoch, acc_test, loss_test))
            # if((epoch + 1) % config.gap_epoch_save_checkpoint == 0):
                # model.cpu()
                # save_pkl(model,join(checkpoints_dir,'%d.pkl'%epoch))
                # model.to(device=device)





        # loss_test = 0.0

    
