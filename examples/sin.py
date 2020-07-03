import numpy as np
import torch
import torch.optim as optim
from kernel_machines.kernel_network import KMClassifier, KernelMachine, MLP
from kernel_machines.utils import train, test, loss, count_parameters, train_bfgs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set("paper")
sns.set_palette("colorblind")
sns.set_style("white")

def generate_data(x_lim, test = False):
    if test:
        torch.manual_seed(1)
    else:
        torch.manual_seed(0)
    xs = 2 * x_lim * torch.rand(100,1) - x_lim
    if test:
        torch.manual_seed(3)
    else:
        torch.manual_seed(2)
    torch.manual_seed(2)
    ys = torch.sin(xs) + torch.rand(100,1)
    return xs, ys


def train_model(model, xs, ys, xs_test, ys_test, lr, num_epochs, cost=0):
    loss_per_epoch = []
    loss_per_epoch_test = []
    # optimizer = optim.Adam(model.parameters(),  lr=lr)

    for epoch in range(num_epochs):
        # train_bfgs(model, torch.cat([xs, ys], 1), zs, device,
        #           loss_func=loss, cost=cost)
        l = train_bfgs(model, xs, ys, device, loss_func=loss, cost=cost)
        loss_per_epoch.append(l.item())
        l_test, _ = test(model, xs_test, ys_test, device, loss_func=loss, cost=cost)
        loss_per_epoch_test.append(l_test.item())
        if epoch % 1000 == 0:
            print("epoch {} loss {}".format(epoch+1, l.item()))

    return model, loss_per_epoch, loss_per_epoch_test


def plot_preds_comparison(xs, ys, km_class, mlp, mlp_sigmoid, ax = None):
    if ax is None: f, ax = plt.subplots()
    order = xs.argsort(axis=0)
    ax.scatter(np.squeeze(xs.detach().numpy())[order],
                np.squeeze(ys.detach().numpy())[order], marker="o")

    xxs = np.linspace(-x_lim, x_lim, 1000).astype(np.float32)

    predmlp = np.squeeze(mlp(torch.from_numpy(np.expand_dims(xxs,1)))[0].detach().numpy())
    predmlp_sigmoid = np.squeeze(mlp_sigmoid(torch.from_numpy(np.expand_dims(xxs,1)))[0].detach().numpy())
    predker = np.squeeze(km_class(torch.from_numpy(np.expand_dims(xxs,1)))[0].detach().numpy())
    truth = np.sin(xxs)+0.5
    ax.plot(xxs, predker, label="km")
    ax.plot(xxs, predmlp, label="mlp", alpha=.5)
    ax.plot(xxs, predmlp_sigmoid, label="mlp_sigmoid", alpha=.5)
    ax.plot(xxs,truth, label="truth")
    ax.legend()
    ax.axis("off")


def plot_losses_comparison(loss_km, loss_mlp, loss_mlp_sigmoid, ax = None, dashed = False):
    if ax is None: f, ax = plt.subplots()
    custom_palette = sns.color_palette("colorblind", 3)
    linestyle = "solid" if not dashed else "dashed"
    subscript = "" if not dashed else " test"
    ax.plot(loss_km, label="km loss" + subscript, linestyle = linestyle, color = custom_palette[0])
    ax.plot(loss_mlp, label="mlp loss" + subscript, linestyle = linestyle, color = custom_palette[1])
    ax.plot(loss_mlp_sigmoid, label="mlp-sigmoid loss" + subscript, linestyle = linestyle, color = custom_palette[2])
    # ax.set_yscale("log")
    ax.legend()
    sns.despine()


if __name__ == "__main__":
    plt.ion()
    device = torch.device("cpu")
    x_lim = 10
    xs, ys = generate_data(x_lim, test = False)
    xs_test, ys_test = generate_data(x_lim, test = True)
    cost = .003
    lr = 0.001
    num_epochs = 3000
    models = []
    losses = []
    #Kernel machine
    km = KernelMachine((2,2,2,1), xs).to(device)
    km_class, km_loss, km_loss_test = train_model(km, xs, ys, xs_test,
                                                  ys_test, lr, num_epochs,
                                                  cost=cost)
    models.append(km_class)
    losses.append(km_loss)
    #Multilayer perceptron with ReLUs
    mlp = MLP((1,16,32), (16,32,1),
              activations = ["relu", "relu", "linear"])
    mlp, mlp_loss, mlp_loss_test = train_model(mlp, xs, ys, xs_test, ys_test,
                                               lr, num_epochs, cost=cost)
    #multilayer perceptron with sigmoids
    mlp_sigmoid = MLP((1,16,32), (16,32,1),
                      activations = ["sigmoid", "sigmoid", "linear"])
    mlp_sigmoid, mlp_sigmoid_loss, mlp_sigmoid_loss_test = train_model(mlp_sigmoid,
                                                                       xs, ys,
                                                                       xs_test,
                                                                       ys_test,
                                                                       lr,
                                                                       num_epochs,
                                                                       cost=cost)
    f, axs = plt.subplots(1,2)
    plot_preds_comparison(xs, ys, km_class, mlp, mlp_sigmoid, ax = axs[0])
    plot_losses_comparison(km_loss, mlp_loss, mlp_sigmoid_loss, ax = axs[1])
    plot_losses_comparison(km_loss_test, mlp_loss_test, mlp_sigmoid_loss_test, ax = axs[1], dashed=True)

    # # print("skm {} mlp {} mlp_sigmoid {}".format(loss_km[-1],
    # #                                             loss_mlp[-1],
    # #                                             loss_mlp_sigmoid[-1]))
    # #
    # #
    # #
    # #
    # #
    # #
    # # print("truth mlp sigmoid norm ", np.linalg.norm(truth-predmlp_sigmoid))
    # # print("truth mlp norm ", np.linalg.norm(truth-predmlp))
    # # print("truth skm norm ", np.linalg.norm(truth-predker))
