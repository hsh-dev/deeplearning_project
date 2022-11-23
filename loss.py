import torch
import torch.nn.functional as F


def discriminator_loss(pred, target: bool = True):
    if target:
        label = torch.ones_like(pred)
    else:
        label = torch.zeros_like(pred)

    loss = F.binary_cross_entropy(torch.sigmoid(pred), label)
    return loss


def content_loss(input, output):
    loss = F.mse_loss(input, output)

    return loss


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss