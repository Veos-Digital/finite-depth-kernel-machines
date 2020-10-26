import torch
import torch.nn as nn


def test_symmetry(a):
    return torch.allclose(a.transpose(0, 1), a)


def radialkernel(u, v):
    uu = torch.pow(u, 2).sum(1, keepdim=True)
    vv = torch.pow(v, 2).sum(1, keepdim=True)
    uv = torch.matmul(u, torch.t(v))
    res = uv - uu/2 - torch.t(vv)/2
    return torch.exp(res)


class KernelMachine(nn.Module):
    def __init__(self, sizes, data, kernel=radialkernel):
        super(KernelMachine, self).__init__()
        self.data = data
        self.data.requires_grad = False
        self.augmenter = nn.Linear(data.shape[1], sum(sizes))
        self.sizes = sizes
        self.num_samples = data.shape[0]
        self.kernel = kernel
        self.c_ss = nn.Parameter(torch.empty(self.num_samples,
                                             sum(self.sizes[1:])),
                                 requires_grad=True)
        nn.init.xavier_uniform_(self.c_ss, gain=1.0)

    def split_xs(self, inputs):
        return list(torch.split(inputs, self.sizes, dim=1))

    def split_cs(self):
        return list(torch.split(self.c_ss, self.sizes[1:], dim=1))

    def forward(self, inputs):
        x = self.augmenter(torch.cat([self.data, inputs], dim=0))
        reg = square_params(self.augmenter)
        xss, css = self.split_xs(x), self.split_cs()
        ker = False
        num_samples = self.num_samples

        for (i, (xs, cs)) in enumerate(zip(xss, css)):
            ker = ker + radialkernel(xs, xs[:num_samples, :])
            val = torch.matmul(ker, cs)
            xss[i+1] = xss[i+1] + val
            reg += torch.sum(cs * val[:num_samples, :])

        return xss[-1][num_samples:, :], reg


def square_params(layer):
    flat_w = layer.weight.view(-1)
    acc = torch.dot(flat_w, flat_w)
    if layer.bias is not None:
        acc += torch.dot(layer.bias, layer.bias)
    return acc


class KMClassifier(nn.Module):
    def __init__(self, kernel_network, output_size):
        super(KMClassifier, self).__init__()
        self.kernel_network = kernel_network
        self.output_layer = nn.Linear(sum(kernel_network.sizes), output_size)
        self.machines = nn.ModuleList([self.kernel_network, self.output_layer])

    def forward(self, input):
        x = input
        reg = 0

        for m in self.machines:
            x, r = m(x)
            reg += r

        return x, reg


class MLP(nn.Module):
    def __init__(self, in_sizes, out_sizes, activations):
        super(MLP, self).__init__()
        self.in_sizes = in_sizes
        self.out_sizes = out_sizes
        self.activations = activations
        self.activations_dict = nn.ModuleDict([['relu', nn.ReLU()],
                                               ['sigmoid', nn.Sigmoid()],
                                               ['linear', nn.Identity()]])
        self.layers = self.build()


    def build(self):
        params = zip(self.in_sizes, self.out_sizes, self.activations)
        layers = nn.ModuleList([])

        for inp, out, act in params:
            layers.append(nn.Linear(inp, out, bias = True))
            layers.append(self.activations_dict[act])

        return layers

    def forward(self, inputs):
        x = inputs
        reg = 0

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if hasattr(layer, "weight"):
                reg += square_params(layer)

        return x, reg
