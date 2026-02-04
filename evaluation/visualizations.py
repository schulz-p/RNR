import torch
import matplotlib.pyplot as plt

from utils import general, coordinate_tensor


def show_images(model, resolution=[100, 100, 100], dim=0, title_experiment="", tag="Fixed"):
    """ Visualizes slice of fixed, moving or difference images
    before or after registration depending on 'tag'.
    The direction of the slice is specified by 'dim'.
    Options for 'tag':
        [Fixed, Moving, Difference, Moving_transformed, Difference_transformed] """

    # Get initial grid and number of time steps
    xc = coordinate_tensor.make_coordinate_tensor(dims=resolution, gpu=model.gpu)
    steps = model.n_layers

    # Check whether intermediate fixed images are used
    intermediate_fixed_images = (model.fixed_image.shape[0] > 1)

    # Iterate over time steps
    for j in range(steps):

        # Get images on grids
        match tag:
            case "Fixed":
                if not intermediate_fixed_images and (j > 0):
                    return
                if j == 0:
                    title_experiment = title_experiment + "fixed"
                if intermediate_fixed_images:
                    title = title_experiment + "_n=" + str(j + 1)
                else:
                    title = title_experiment
                image_fixed = general.fast_trilinear_interpolation(
                    model.fixed_image[j], xc[:, 0], xc[:, 1], xc[:, 2],
                )
                image = image_fixed

            case "Moving":
                if j > 0:
                    return
                title = title_experiment + "moving"
                image_moving = general.fast_trilinear_interpolation(
                    model.moving_image, xc[:, 0], xc[:, 1], xc[:, 2],
                )
                image = image_moving

            case "Difference":
                if not intermediate_fixed_images and (j > 0):
                    return
                if j == 0:
                    title_experiment = title_experiment + "difference"
                if intermediate_fixed_images:
                    title = title_experiment + "_n=" + str(j + 1)
                else:
                    title = title_experiment

                image_fixed = general.fast_trilinear_interpolation(
                    model.fixed_image[j], xc[:, 0], xc[:, 1], xc[:, 2],
                )
                image_moving = general.fast_trilinear_interpolation(
                    model.moving_image, xc[:, 0], xc[:, 1], xc[:, 2],
                )
                image = image_fixed - image_moving

            case "Moving_transformed":
                if j == 0:
                    title_experiment = title_experiment + "moving_transformed"
                title = title_experiment + "_n=" + str(j + 1)

                # Get deformed grid
                with torch.no_grad():
                    yc = model.network(xc, steps=j + 1)
                image_moving_transformed = general.fast_trilinear_interpolation(
                    model.moving_image, yc[:, 0], yc[:, 1], yc[:, 2],
                )
                image = image_moving_transformed

            case "Difference_transformed":
                if j == 0:
                    title_experiment = title_experiment + "difference_transformed"
                title = title_experiment + "_n=" + str(j + 1)

                if intermediate_fixed_images:
                    image_fixed = general.fast_trilinear_interpolation(
                        model.fixed_image[j], xc[:, 0], xc[:, 1], xc[:, 2],
                    )
                else:
                    image_fixed = general.fast_trilinear_interpolation(
                        model.fixed_image[0], xc[:, 0], xc[:, 1], xc[:, 2],
                    )
                # Get deformed grid
                with torch.no_grad():
                    yc = model.network(xc, steps=j + 1)
                image_moving_transformed = general.fast_trilinear_interpolation(
                    model.moving_image, yc[:, 0], yc[:, 1], yc[:, 2],
                )
                image = image_fixed - image_moving_transformed

            case _:
                print("Unknown tag")
                return

        # Reshape and permute image such that correct image slice can be extracted
        image = torch.reshape(image, resolution).cpu().detach()
        match dim:
            case 0:
                d = (0, 1, 2)
            case 1:
                d = (1, 0, 2)
            case 2:
                d = (2, 1, 0)
            case _:
                print("d<3 not fulfilled")
                return
        image = torch.permute(image, d)

        # Select center slice
        slice = int(resolution[dim] / 2)
        image_slice = image[slice, :, :]

        # Plot and save image
        plt.figure(figsize=(5, 5))
        plt.imshow(image_slice, vmin=-1500, vmax=1500, cmap="gray")
        plt.axis("off")
        plt.axis("equal")
        plt.savefig("results/" + title + "_dim=" + str(dim))
        plt.title(title + "_dim=" + str(dim))
        plt.show()


def plot_loss(model, title_experiment=""):
    """ Plots total loss and data fitting term over epochs."""
    plt.plot(
        [i * model.log_interval for i in range(len(model.loss_list))],
        model.loss_list,
        label="loss"
    )
    plt.plot(
        [i * model.log_interval for i in range(len(model.data_loss_list))],
        model.data_loss_list,
        label="data loss"
    )
    plt.legend(loc="best")
    plt.savefig("results/" + title_experiment + 'loss')
    plt.title("Loss vs epochs")
    plt.show()
