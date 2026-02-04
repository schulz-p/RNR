import numpy as np
import torch

from utils.general import jacobian_matrix


def ratio_convolutions_coordinates(input_coords, output_coords):
    """ Computes percentage of convolution given input and output coordinates for a deformation. """

    # Compute determinants of Jacobian matrices
    jac = jacobian_matrix(input_coords, output_coords)
    det_jac = torch.det(jac)

    num_conv = sum(det_jac<=0).item()
    percentage_conv = num_conv / input_coords.size(0)

    return percentage_conv


def ratio_convolutions_deformation(model, num_input_coords=4E6):
    """ Computes percentage of convolution caused by the deformation given by 'model'.
     Here 'num_input_coords' randomly chosen coordinates are considered. """

    # Choose coordinates
    indices = torch.randperm(model.possible_coordinate_tensor.shape[0], device=model.device)[: int(num_input_coords)]
    xc = model.possible_coordinate_tensor[indices, :]
    xc = xc.requires_grad_(True)

    # Deform coordinates
    yc = model.network(xc)

    # Calculate percentage of convolution
    ratio_conv = ratio_convolutions_coordinates(xc, yc)
    return ratio_conv


def landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    """ Computes mean and standard deviation of landmark errors.
    Format of outputs: [2-norm, dist in x_1-dir, dist in x_2-dir, dist in x_3-dir].
    Implementation from the GitHub repository https://github.com/MIAGroupUT/IDIR. """

    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds


def deform_landmarks(model, landmarks_pre, image_size):
    """ Computes transformed landmarks (applies deformation on original landmarks).
     Implementation from the GitHub repository https://github.com/MIAGroupUT/IDIR with minor adaptations. """

    scale_of_axes = [(0.5 * s) for s in image_size]

    coordinate_tensor = torch.FloatTensor(landmarks_pre / scale_of_axes) - 1.0
    if model.gpu:
        coordinate_tensor = coordinate_tensor.cuda()

    landmarks_new = model.network(coordinate_tensor)

    landmarks_new = (landmarks_new.cpu().detach().numpy()+1.0) * scale_of_axes
    delta = landmarks_new - landmarks_pre

    return landmarks_new, delta
