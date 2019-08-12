import torch as th
import torch.nn.functional as F


def loss_function(recon_x, x, mu, log_var):
    """
    :param recon_x: reconstructed input
    :param x: input
    :param mu: parameter of posterior (variational parameter)
    :param log_var: parameter of posterior (variational parameter)
    :return: VAE loss
    """

    # reconstruction loss
    reconstruction_loss = F.binary_cross_entropy(recon_x, x)

    # KL loss
    kl_loss = 1 + log_var - mu.pow(2) - log_var.exp()
    kl_loss = th.sum(kl_loss, dim=-1)
    kl_loss *= -0.5

    return th.mean(reconstruction_loss + kl_loss)
