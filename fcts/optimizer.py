import torch as th
import torch.nn.functional as F


def loss_function(recon_x, x, mu, log_var, factors=False, weight=None):
    """
    :param recon_x: reconstructed input
    :param x: input
    :param mu: parameter of posterior (variational parameter)
    :param log_var: parameter of posterior (variational parameter)
    :param factors: use or not use additional factors for the KL and BCE terms (default=False)
    :param weight: weight matrix used in BCE term (default=None)
    :return: VAE loss
    """
    # reconstruction loss
    if weight is not None:
        reconstruction_loss = F.binary_cross_entropy(recon_x.view(-1), x.view(-1), weight=weight)
    else:
        reconstruction_loss = F.binary_cross_entropy(recon_x, x)

    # KL loss
    kl_loss = 1 + log_var - mu.pow(2) - log_var.exp()
    kl_loss = th.sum(kl_loss, dim=-1)
    kl_loss *= -0.5

    if factors:
        norm = x.size()[0] * x.size()[0] / float((x.size()[0] * x.size()[0] - x.sum()) * 2)
        kl_loss *= 1./x.size()[0]
        reconstruction_loss *= norm

    return th.mean(reconstruction_loss) + th.mean(kl_loss)
