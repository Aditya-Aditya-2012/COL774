import torch
import torch.nn.functional as F

def smooth_crossentropy(pred, gold, smoothing=0.1):
    return F.cross_entropy(pred, gold, label_smoothing=smoothing)
