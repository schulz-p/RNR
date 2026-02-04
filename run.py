import torch

from data import data_loader
from models import model
from evaluation import visualizations, evaluation_measures


""" Initialization  """

# Initialize paths
dir = r"/RNR_Dirlab"
data_dir = dir + r"/data"
out_dir = dir + r"/results"
network_path = out_dir + r"/network.npz"

# Choose data case
case_id = 2

# Load data for specified case
(
    img_insp,
    img_exp,
    landmarks_insp,
    landmarks_exp,
    mask_exp,
    voxel_size,
    ) = data_loader.load_data_DIRLab(
    folder="{}/Case".format(data_dir), case_id=case_id, include_intermediate=True
)

# Set parameters other than default (see model.set_default_arguments() for default parameters)
args = {}
args["mask"] = mask_exp
args["batch_size"] = int(sum(mask_exp.flatten() > 0) / 30)  # use approx. 3% of voxels in each epoch
args["epochs"] = 300
args["gpu"] = False

learn = True    # True: network trained, False: network loaded

# Initialize object handling registration
ImpReg = model.ImplicitRegistrator(img_exp, img_insp, **args)


""" Visualize initial setting """
title_experiment = "case" + str(case_id) + "/"
resolution = ImpReg.moving_image.shape
# Visualize fixed images, moving image before registration and corresponding difference images
visualizations.show_images(
    ImpReg, resolution=resolution, dim=1, title_experiment=title_experiment, tag="Fixed"
)
visualizations.show_images(
    ImpReg, resolution=resolution, dim=1, title_experiment=title_experiment, tag="Moving"
)
visualizations.show_images(
    ImpReg, resolution=resolution, dim=1, title_experiment=title_experiment, tag="Difference"
)


""" Registration """
# Learn and save network, which represents the deformation
if learn:
    ImpReg.fit()
    torch.save(ImpReg.network.state_dict(), network_path)
# Load network, which represents the deformation
ImpReg.network.load_state_dict(torch.load(network_path))


""" Validation """
# Calculate landmark accuracy
landmarks_new, _ = evaluation_measures.deform_landmarks(
    ImpReg, landmarks_insp, image_size=img_exp.shape
)
accuracy_mean, accuracy_std = evaluation_measures.landmark_accuracy(
    landmarks_new, landmarks_exp, voxel_size=voxel_size
)
print("LM accuracy mean:", accuracy_mean)  # format: [2-norm, dist in x_1-dir, dist in x_2-dir, dist in x_3-dir]
print("LM accuracy std:", accuracy_std)    # format: [2-norm, dist in x_1-dir, dist in x_2-dir, dist in x_3-dir]

# Check whether folding of the image occurs
conv = evaluation_measures.ratio_convolutions_deformation(ImpReg, num_input_coords=1E5) # percentage of convolutions
if conv > 0:
    print("Image folding occurs")
else:
    print("No image folding occurs")

# Visualize for each time step resulting deformed image and corresponding difference image
visualizations.show_images(
    ImpReg, resolution=resolution, dim=1, title_experiment=title_experiment, tag="Moving_transformed"
)
visualizations.show_images(
    ImpReg, resolution=resolution, dim=1, title_experiment=title_experiment, tag="Difference_transformed"
)

# Visualize loss over epochs
visualizations.plot_loss(ImpReg, title_experiment=title_experiment)
