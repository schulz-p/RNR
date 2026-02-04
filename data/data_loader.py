import os
import SimpleITK as sitk
import numpy as np
import torch


def load_data_DIRLab(folder, case_id=1, include_intermediate=False):
    """ Loads data for one DIR-Lab case.
    data: inspiration image, expiration image, optionally intermediate images, corresponding landmarks, data mask
    Implementation from the GitHub repository https://github.com/MIAGroupUT/IDIR with adaptations. """

    # Size of data per image pair
    image_sizes = [
        0,
        [94, 256, 256],
        [112, 256, 256],
        [104, 256, 256],
        [99, 256, 256],
        [106, 256, 256],
        [128, 512, 512],
        [136, 512, 512],
        [128, 512, 512],
        [128, 512, 512],
        [120, 512, 512],
    ]

    # Scale of data per image pair
    voxel_sizes = [
        0,
        [2.5, 0.97, 0.97],
        [2.5, 1.16, 1.16],
        [2.5, 1.15, 1.15],
        [2.5, 1.13, 1.13],
        [2.5, 1.1, 1.1],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
    ]

    shape = image_sizes[case_id]

    # Load images and mask
    folder = folder + str(case_id) + r"Pack" + os.path.sep
    dtype = np.dtype(np.int16)

    if include_intermediate:
        file_insp = []
        for k in range(4,-1,-1):
            file_insp.append(folder + r"Images/case" + str(case_id) + "_T" + str(k) + "0")
    else:
        file_insp = [folder + r"Images/case" + str(case_id) + "_T00"]
    file_exp = folder + r"Images/case" + str(case_id) + "_T50"
    file_mask = folder + r"Masks/case" + str(case_id) + "_T00_mask.mhd"

    # Handle different naming conventions of the dataset
    if case_id == 1:
        file_insp_end = "_s.img"
        file_exp_end = "_s.img"
    elif case_id >= 2 and case_id <= 5:
        file_insp_end = "-ssm.img"
        file_exp_end = "-ssm.img"
    else:
        file_insp_end = ".img"
        file_exp_end = ".img"

    image_insp = np.zeros((len(file_insp), shape[0], shape[1], shape[2]))
    for k in range(len(file_insp)):
        with open(file_insp[k] + file_insp_end, "rb") as f:
            data = np.fromfile(f, dtype)
        image_insp[k] = data.reshape(shape)

    with open(file_exp + file_exp_end, "rb") as f:
        data = np.fromfile(f, dtype)
    image_exp = data.reshape(shape)

    imgsitk_in = sitk.ReadImage(file_mask)
    mask = np.clip(sitk.GetArrayFromImage(imgsitk_in), 0, 1)

    image_insp = torch.FloatTensor(image_insp)
    image_exp = torch.FloatTensor(image_exp)

    # Load landmarks
    if case_id <= 5:
        file_landmarks = folder + r"ExtremePhases/Case" + str(case_id) + "_300_"
    else:
        file_landmarks = folder + r"extremePhases/case" + str(case_id) + "_dirLab300_"

    with open(file_landmarks + "T00_xyz.txt") as f:
        landmarks_insp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    with open(file_landmarks + "T50_xyz.txt") as f:
        landmarks_exp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    landmarks_insp[:, [0, 2]] = landmarks_insp[:, [2, 0]]
    landmarks_exp[:, [0, 2]] = landmarks_exp[:, [2, 0]]

    return (
        image_insp,
        image_exp,
        landmarks_insp,
        landmarks_exp,
        mask,
        voxel_sizes[case_id],
    )
