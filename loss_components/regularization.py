import torch

from utils import general

def grad_velocity_penalty(input_coords, velocity):
    """ penalty: Computes norm of gradients of the velocities (residual functions).
    By adding this to the loss the smoothness of the deformation is controlled and
    diffeomorphic solutions are preferred. """

    loss = 0
    for v in velocity:
        jac = general.jacobian_matrix(input_coords, v)
        loss += torch.sum(torch.square(torch.linalg.matrix_norm(jac, dim=(1,2))))

    return loss / (len(input_coords)*len(velocity))


def weight_decay_regularization(parameters):
    """ regularization: Computes norm of weight matrices in the model.
    Adding this to loss ensures existence of solutions.
    Note: The bias vectors are not included here since these are naturally bounded
    due to the registration problem and the optimization. """

    loss = 0
    num_param = 0

    for param in parameters:
        if len(param.shape) == 2:
            loss = loss + torch.norm(param) ** 2
            num_param += param.size()[0] * param.size()[1]
    loss = loss / num_param
    return loss
