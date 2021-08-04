import torch
import torch.nn as nn
from config import CHECKPOINT_DIR
import os

def weights_init_unif(module, a, b):
    """
    Initialize all weights in module to values within [a, b].

    Args:
        module (nn.Module): Target network 
        a (float): Lower bound 
        b (float): Upper bound
    """

    for p in module.parameters():
        nn.init.uniform_(p.data, a=a, b=b)

def load_from_checkpoint(model, optimizer, activation, lr, epoch, device, dataset):

    # To start from epoch e, we need to load in a pretrained model up to epoch e-1

    checkpoint_name = f"{dataset}-activation-{activation}_LR-{lr}_epoch-{epoch-1}.pt"

    path = None
    for root, dirs, files in os.walk(CHECKPOINT_DIR):
        for filename in files:
            if filename == checkpoint_name:
                path = os.path.join(root, filename)

    if not path:
        raise FileNotFoundError("Desired checkpoint does not exist")
    else:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return model, optimizer

def save_checkpoint(model, optimizer, epoch, activation, lr, dataset):

    dirname = f"{dataset}-activation-{activation}_LR-{lr}"
    filename = f"{dataset}-activation-{activation}_LR-{lr}_epoch-{epoch}.pt"

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, dirname)
    save_path = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, save_path)