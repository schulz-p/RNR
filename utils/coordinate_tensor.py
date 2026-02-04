import numpy as np
import torch


def make_coordinate_tensor(mask=None, dims=(28, 28, 28), gpu=True):
    """ Produces tensor with images coordinates.
    Implementation from the GitHub repository https://github.com/MIAGroupUT/IDIR with minor adaptations. """

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    if mask is not None:
        coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    if gpu:
        coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor
