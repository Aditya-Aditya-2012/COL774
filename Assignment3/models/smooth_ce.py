import torch
import torch.nn.functional as F

def smooth_crossentropy(pred, gold, smoothing=0.1):
    # If using PyTorch 1.10 or later
    return F.cross_entropy(pred, gold, label_smoothing=smoothing)

    # If using an earlier version of PyTorch, use the adjusted manual implementation
    # n_class = pred.size(1)
    # smoothing_value = smoothing / (n_class - 1)
    # with torch.no_grad():
    #     one_hot = torch.full_like(pred, smoothing_value)
    #     one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    # log_prob = F.log_softmax(pred, dim=1)
    # loss = F.kl_div(input=log_prob, target=one_hot, reduction='batchmean')
    # return loss
