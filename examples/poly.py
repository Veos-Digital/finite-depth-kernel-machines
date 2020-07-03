import sys
sys.path.append('../')
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from kernel_machines.kernel_network import KMClassifier, KernelMachine, MLP
from kernel_machines.utils import train, test, loss, count_parameters,train_bfgs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
sns.set("paper")
sns.set_palette("colorblind")
sns.set_style("white")

def f(x,y):
    return (2 * x - 1)**2 + 2 * y + x * y - 3


def plot_surfaces(models, training_data = None, model_names = []):
    x = np.linspace(0,1,num=100)
    y = np.linspace(0,1,num=100)
    xx, yy = np.meshgrid(x, y)
    truth = f(xx, yy)
    zzs = []
    xx = torch.from_numpy(xx).float()
    yy = torch.from_numpy(yy).float()
    truth = torch.from_numpy(truth).float()

    for model in models:
        zz = []

        for x, y, t in zip(xx,yy, truth):
            _, pr = test(model, torch.cat([x.unsqueeze(1),y.unsqueeze(1)], 1),
                         t.unsqueeze(1), device, loss_func = loss, cost=0)
            zz.append(pr)
        zz = torch.cat(zz,1)
        zzs.append(zz.T)

    fig, axs = plt.subplots(1,1 + len(models), subplot_kw=dict(projection='3d'))
    titles = ["ground truth"] + model_names

    for i, (vals, ax) in enumerate(zip([truth] + zzs, axs)):
        ax.plot_surface(xx.numpy(), yy.numpy(), vals.numpy(), rstride=1, cstride=1,
                cmap='Blues', edgecolor='none')
        ax.set_zlim([-3,1])
        if training_data is not None and i == 0:
            ax.scatter(training_data[0][:,0].numpy(),
                       training_data[1][:,0].numpy(),
                       training_data[2][:,0].numpy())
        ax.set_title(titles[i])
        ax.axis("off")


def get_data(num_samples, func = f, train = True, grid = False):
    if grid and train:
        x, y = torch.linspace(0,1,6), torch.linspace(0,1,6)
        xx,yy = torch.meshgrid(x,y)
        xs, ys = xx.reshape(-1,1), yy.reshape(-1,1)
        return xs, ys, func(xs, ys)
    seeds = [0, 1] if train else [3,4]
    N = num_samples
    torch.manual_seed(seeds[0])
    xs = torch.rand(N,1)
    torch.manual_seed(seeds[1])
    ys = torch.rand(N,1)
    return xs, ys, func(xs, ys)


def train_and_test(training_data, testing_data, device, model, lr,
                   num_samples = 100):
    N = num_samples
    xs, ys, zs = training_data
    xs_test, ys_test, zs_test = testing_data
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_epochs = 5000
    loss_per_epoch = []
    loss_per_epoch_test = []
    cost = 0

    for epoch in range(num_epochs):
        l = train_bfgs(model, torch.cat([xs, ys], 1), zs, device,
                  loss_func=loss, cost=cost)
        # l = train(model, torch.cat([xs, ys], 1), zs, optimizer, device,
        #           loss_func=loss, cost=cost)
        l_test, _ = test(model, torch.cat([xs_test, ys_test], 1), zs_test, device,
                      loss_func = loss, cost=cost)
        if epoch % 100 == 0:
            print("epoch {} loss {}".format(epoch + 1, l.item()))
            print("epoch {} loss test {}".format(epoch + 1, l_test.item()))
        loss_per_epoch.append(l.item())
        loss_per_epoch_test.append(l_test.item())

    return model, loss_per_epoch, loss_per_epoch_test


if __name__ == "__main__":
    plt.ion()
    device = torch.device("cpu")
    num_samples = 50
    lr = 0.003
    training_data = get_data(num_samples, func = f, train=True, grid=True)
    testing_data = get_data(num_samples, func = f, train=False)
    rad_km = KernelMachine((2, 3, 2, 1),
                           torch.cat(training_data[:-1], dim=1)).to(device)
    rad_km, rad_loss, rad_test_loss = train_and_test(training_data, testing_data,
                                                     device, rad_km, lr,
                                                     num_samples = num_samples)

    mlp = MLP((2,16,16), (16,16,1), activations = ["relu", "relu", "linear"])
    mlp, mlp_loss, mlp_test_loss = train_and_test(training_data, testing_data,
                                                  device, mlp, lr,
                                                  num_samples = num_samples)

    names= ["rad_km", "mlp"]
    plot_surfaces([rad_km, mlp], training_data = training_data, model_names = names)
    f,a = plt.subplots()
    losses = [rad_loss, mlp_loss]
    test_losses = [rad_test_loss, mlp_test_loss]
    custom_palette = sns.color_palette("colorblind", len(losses))

    for l, tl, n, c in zip(losses[::-1], test_losses[::-1], names[::-1], custom_palette):
        a.plot(l, label = n, color = c)
        a.plot(tl, linestyle = "dashed", color = c)
    a.set_yscale("log")
    a.legend()
    sns.despine()
