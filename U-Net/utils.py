import torch
from prepare_data import *

def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save(model, model_path):
    torch.save({
        'model_state_dict': model.state_dict,
    }, model_path)
