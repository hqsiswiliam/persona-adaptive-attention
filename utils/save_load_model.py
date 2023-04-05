import torch

from utils.eprint import eprint


def save_model(path, model, optimizer, epoch, config, loss):
    if not config.training.save_model:
        eprint("Warning, epoch: {}, the model will not be saved!!!!".format(epoch))
        return
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': loss
    }, path)


def load_model(model, path, optimizer=None, device='cuda', strict=True):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss
