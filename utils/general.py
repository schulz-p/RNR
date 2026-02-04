import torch


def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    """ Trilinear interpolation:
    Implementation from the GitHub repository https://github.com/MIAGroupUT/IDIR. """

    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output


def jacobian_matrix(input_coords, output):
    """ Computes the Jacobian matrix of 'output' wrt 'input_coords'.
    Implementation from the GitHub repository https://github.com/MIAGroupUT/IDIR with minor adaptations. """

    jac = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        jac[:, i, :] = gradient(input_coords, output[:, i])
    return jac


def gradient(input_coords, output):
    """ Computes gradient of 'output' wrt 'input_coords'.
    Implementation from the GitHub repository https://github.com/MIAGroupUT/IDIR. """

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True, allow_unused=True
    )[0]
    return grad
