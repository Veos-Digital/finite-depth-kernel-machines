import numpy as np
import torch
import torch.optim as optim
from kernel_machines.kernel_network import KMClassifier, KernelMachine, MLP
from kernel_machines.utils import train, loss, count_parameters, train_bfgs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set("paper")
sns.set_palette("colorblind")
sns.set_style("white")

def generate_data(x_lim):
    torch.manual_seed(0)
    xs = 2 * x_lim * torch.rand(100,1) - x_lim
    torch.manual_seed(2)
    ys = torch.sin(xs) + torch.rand(100,1)
    return xs, ys


def train_model(model, lr, num_epochs, cost=0):
    loss_per_epoch = []
    loss_test_per_epoch = []

    for epoch in range(num_epochs):
        l = train_bfgs(model, xs, ys, device, loss_func=loss, cost=cost)
        loss_per_epoch.append(l.item())
        if epoch % 1000 == 0:
            print("epoch {} loss {}".format(epoch+1, l.item()))

    return model, loss_per_epoch


def plot_preds_costs(xs, ys, models, costs, ax = None):
    if ax is None: f, ax = plt.subplots()
    order = xs.argsort(axis=0)
    ax.scatter(np.squeeze(xs.detach().numpy())[order],
                np.squeeze(ys.detach().numpy())[order], marker="o")
    xxs = np.linspace(-x_lim, x_lim, 1000).astype(np.float32)
    for m, c in zip(models, costs):
        pred = np.squeeze(m(torch.from_numpy(np.expand_dims(xxs,1)))[0].detach().numpy())
        ax.plot(xxs, pred, label=str(c))
    truth = np.sin(xxs)+0.5
    ax.plot(xxs,truth, label="truth")
    ax.legend()
    ax.axis("off")


def plot_losses_costs(losses, costs, ax = None):
    if ax is None: f, ax = plt.subplots()
    for l, c in zip(losses, costs): ax.plot(l, label=str(c))
    ax.legend()
    sns.despine()



if __name__ == "__main__":
    plt.ion()
    device = torch.device("cpu")
    x_lim = 10
    xs, ys = generate_data(x_lim)
    costs = [.1, .01, .003]
    lr = 0.001
    num_epochs = 15000
    models = []
    losses = []
    #Kernel machine
    for cost in costs:
        km_class = KernelMachine((2,2,2,1), xs).to(device)
        km_class, km_loss = train_model(km_class, lr, num_epochs, cost=cost)
        models.append(km_class)
        losses.append(km_loss)

    f, axs = plt.subplots(1,2)
    plot_preds_costs(xs, ys, models, costs, ax = axs[0])
    plot_losses_costs(losses, costs, ax = axs[1])
