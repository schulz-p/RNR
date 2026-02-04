import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from utils import general, coordinate_tensor
from models import network
from loss_components import ncc, regularization


""" Enables registration with implicit neural representation (INR).
As code basis for this class 'models.ImplicitRegistrator' from the 
GitHub repository https://github.com/MIAGroupUT/IDIR was used, but 
significant model adjustments have been made. """
class ImplicitRegistrator:

    def __init__(self, moving_image, fixed_image, **args):

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()

        # Check if all keys in args are valid (this checks for typos)
        assert all(arg in self.args.keys() for arg in args)

        """ Parse arguments to registrator """
        # Optimization arguments
        self.gpu = args["gpu"] if "gpu" in args else self.args["gpu"]
        self.epochs = args["epochs"] if "epochs" in args else self.args["epochs"]
        self.lr = args["lr"] if "lr" in args else self.args["lr"]
        self.batch_size = args["batch_size"] if "batch_size" in args else self.args["batch_size"]
        self.momentum = args["momentum"] if "momentum" in args else self.args["momentum"]
        self.optimizer = args["optimizer"] if "optimizer" in args else self.args["optimizer"]
        self.log = args["log"] if "log" in args else self.args["log"]
        self.log_interval = args["log_interval"] if "log_interval" in args else self.args["log_interval"]

        # Network and registration arguments
        self.shared_weights = args["shared_weights"] if "shared_weights" in args else self.args["shared_weights"]
        self.n_layers = args["n_layers"] if "n_layers" in args else self.args["n_layers"]
        self.hidden_channels = args["hidden_channels"] if "hidden_channels" in args else self.args["hidden_channels"]
        self.mask = args["mask"] if "mask" in args else self.args["mask"]
        self.image_shape = args["image_shape"]  if "image_shape" in args else self.args["image_shape"]
        self.loss_function = args["loss_function"] if "loss_function" in args else self.args["loss_function"]
        self.grad_velocity_penalty = (
            args["grad_velocity_penalty"]
            if "grad_velocity_penalty" in args
            else self.args["grad_velocity_penalty"]
        )
        self.alpha_grad_velocity = (
            args["alpha_grad_velocity"]
            if "alpha_grad_velocity" in args
            else self.args["alpha_grad_velocity"]
        )
        self.alpha_weight_decay_regularization = (
            args["alpha_weight_decay_regularization"]
            if "alpha_weight_decay_regularization" in args
            else self.args["alpha_weight_decay_regularization"]
        )

        # Make lists to save loss
        self.loss_list = []
        self.data_loss_list = []

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Load network
        self.network = network.ResNet(self.n_layers, self.hidden_channels, self.shared_weights)
        if self.gpu:
            self.network.cuda()

        # Choose optimizer
        if self.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )

        elif self.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        elif self.optimizer.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.lr)

        else:
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer)
                + " not recognized as optimizer, picked SGD instead"
            )

        # Choose loss function
        if self.loss_function.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function.lower() == "ncc":
            self.criterion = ncc.NCC()

        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function)
                + " not recognized as loss function, picked MSE instead"
            )


        # Initialize images and vector with coordinates that will be transformed
        self.moving_image = moving_image
        self.fixed_image = fixed_image

        self.device = "cpu"

        self.possible_coordinate_tensor = coordinate_tensor.make_coordinate_tensor(
            mask=self.mask, dims=self.fixed_image[0].shape,gpu=self.gpu
        )
        indices_random = torch.randperm(self.possible_coordinate_tensor.shape[0], device=None)
        self.possible_coordinate_tensor = self.possible_coordinate_tensor[indices_random, :]

        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()
            self.device = "cuda"


    def cuda(self):
        """ Moves model to GPU."""

        self.network.cuda()
        self.moving_image.cuda()
        self.fixed_image.cuda()


    def set_default_arguments(self):

        self.args = {}

        # Optimization arguments
        self.args["gpu"] = torch.cuda.is_available()
        self.args["epochs"] = 2500
        self.args["lr"] = 1E-4
        self.args["batch_size"] = 10000
        self.args["momentum"] = 0.5
        self.args["optimizer"] = "Adam"
        self.args["log"] = True
        self.args["log_interval"] = 10
        self.args["seed"] = 3

        # Network and registration arguments
        self.args["shared_weights"] = False     # True: velocity field constant in time
        self.args["n_layers"] = 5
        self.args["hidden_channels"] = 100
        self.args["mask"] = None
        self.args["image_shape"] = (200, 200)
        self.args["loss_function"] = "ncc"
        self.args["grad_velocity_penalty"] = True
        self.args["alpha_grad_velocity"] = 1
        self.args["alpha_weight_decay_regularization"] = 1E-4


    def training_iteration(self, epoch):

        # Reset gradient
        self.network.train()

        # Get coordinates used for the training step
        num_coordinates = self.possible_coordinate_tensor.shape[0]
        index_start     = (epoch * self.batch_size) % num_coordinates
        index_end       = (index_start + self.batch_size) % num_coordinates
        if index_start < index_end:
            coordinate_tensor = self.possible_coordinate_tensor[index_start:index_end, :]
        else:
            coordinate_tensor = torch.cat(
                (self.possible_coordinate_tensor[index_start:, :], self.possible_coordinate_tensor[:index_end, :]))
        coordinate_tensor = coordinate_tensor.requires_grad_(True)

        # Calculate data fitting term
        data_loss = 0
        for j in range(len(self.fixed_image)):
            output_j = self.network(coordinate_tensor, steps=self.n_layers - j)
            transformed_image = general.fast_trilinear_interpolation(
                self.moving_image,
                output_j[:, 0],
                output_j[:, 1],
                output_j[:, 2])
            fixed_image = general.fast_trilinear_interpolation(
                self.fixed_image[-(1+j)],
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2])
            data_loss = data_loss + self.criterion(transformed_image, fixed_image)
        loss = 1 / len(self.fixed_image) * data_loss

        # Store values of the data loss
        if self.log:
            if epoch % self.log_interval == 0:
                self.data_loss_list.append(1 / self.log_interval * loss.detach().cpu().numpy())
            else:
                self.data_loss_list[epoch // self.log_interval] += 1 / self.log_interval * loss.detach().cpu().numpy()


        # Add penalty to loss (gradient of velocities)
        output, velocity = self.network(coordinate_tensor, get_velocity=True)
        if self.grad_velocity_penalty:
            loss += self.alpha_grad_velocity * regularization.grad_velocity_penalty(
                coordinate_tensor, velocity
            )

        # Add regularization to loss (this ensures existence of solutions to the minimization problem)
        loss += self.alpha_weight_decay_regularization * regularization.weight_decay_regularization(
            list(self.network.parameters())
        )

        # Store value of the total loss
        if self.log:
            if epoch % self.log_interval == 0:
                self.loss_list.append(1 / self.log_interval * loss.detach().cpu().numpy())
            else:
                self.loss_list[epoch // self.log_interval] += 1 / self.log_interval * loss.detach().cpu().numpy()

        # Perform backpropagation and update the parameters accordingly
        for param in self.network.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()


    def fit(self):
        """ Performs registration by training a network, which parameterizes the deformation. """

        # Perform training iterations
        for i in tqdm.tqdm(range(self.epochs)):
            self.training_iteration(i)
