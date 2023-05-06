import torch


def to_np(tensor):
    """
    Converts a PyTorch tensor to a NumPy array.

    :param tensor: PyTorch tensor to convert.
    :return: NumPy array.
    """
    return tensor.detach().cpu().numpy()


def to_var(x, requires_grad=False):
    """
    Converts a NumPy array to a PyTorch tensor.

    :param x: NumPy array to convert.
    :param requires_grad: Whether the tensor requires gradients.
    :return: PyTorch tensor.
    """
    x = torch.tensor(x, requires_grad=requires_grad)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def weights_init(m):
    """
    Initializes the weights of a PyTorch module.

    :param m: PyTorch module to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def save(net, path):
    """
    Saves a PyTorch module to a file.

    :param net: PyTorch module to save.
    :param path: Path to save the module to.
    """
    torch.save(net.state_dict(), path)


def load(net, path):
    """
    Loads a PyTorch module from a file.

    :param net: PyTorch module to load.
    :param path: Path to load the module from.
    """
    net.load_state_dict(torch.load(path))