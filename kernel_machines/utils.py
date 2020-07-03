import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

"from: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8"
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss(preds, y, cost, reg):
    if cost != 0:
        regularization_cost = cost * reg
    else:
        regularization_cost = 0
    diff = preds - y
    mse = torch.sum(diff * diff) / preds.numel()
    return mse + regularization_cost


def train_bfgs(model, X, y, device, loss_func = loss, cost = .1):

    def closure():
        if torch.is_grad_enabled():
            opt.zero_grad()
        output, reg = model(X_)
        l = loss(output, y_, cost, reg)
        if l.requires_grad:
            l.backward()
        return l

    opt = optim.LBFGS(model.parameters(), max_iter=20, tolerance_grad=1e-07,
                      tolerance_change=1e-09, history_size=100)
    X_ = Variable(X, requires_grad=True)
    y_ = Variable(y)
    opt.step(closure)
    output = model(X_)
    return closure()


def train(model, X, y, optimizer, device, loss_func = None, cost=.1):
    model.train()
    X = X.to(device)
    y = y.to(device)
    pred, reg = model(X)
    l = loss_func(pred, y, cost, reg)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    return l


def test(model, X, y, device, loss_func = None, cost=.1):
    model.eval()

    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        pred, reg = model(X)
        l = loss_func(pred, y, cost, reg)

    return l, pred
