import h5py
import numpy as np


def make_DataLoader(X,y,batch_size,
                    drop_last=False,train_split = 0.8):
    """
    make tensor data loader for training

    Args:
        X: Tensor of features
        y: Tensor of target
        batch_size: Batch size
        drop_last: If drop the last batch which does not have same number of mini batch
        train_split: A ratio of train and validation split 

    Return: 
        train_dl, val_dl: The train and validation DataLoader
    """

    from torch.utils.data import DataLoader, TensorDataset,random_split
    try: 
        dataset = TensorDataset(*X,y)
    except:
        print("The data is not torch.tenor!")

    len_d = len(dataset)
    train_size = int(train_split * len_d)
    valid_size = len_d - train_size

    train_d , val_d = random_split(dataset,[train_size, valid_size])
    
    train_dl = DataLoader(train_d,batch_size=batch_size,drop_last=drop_last,shuffle=True)
    val_dl = DataLoader(val_d,batch_size=batch_size,drop_last=drop_last,shuffle=True)
    
    return train_dl, val_dl