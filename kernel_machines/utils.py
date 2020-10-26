import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def count_parameters(model):
    """from:
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-
    model/4325/8"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse_loss(preds, y, coeff, reg):
    if coeff != 0:
        regularization_cost = coeff * reg
    else:
        regularization_cost = 0
    diff = preds - y
    mse = torch.sum(diff * diff) / preds.numel()
    return mse + regularization_cost


def classification_loss(preds, y, coeff, reg):
    if coeff != 0:
        regularization_cost = coeff * reg
    else:
        regularization_cost = 0
    loss_value = torch.nn.CrossEntropyLoss()(preds, y)
    return loss_value + regularization_cost


def train_bfgs(model, X, y, device, loss_func=None, cost=.1, classify=False):

    def closure():
        if torch.is_grad_enabled():
            opt.zero_grad()
        output, reg = model(X_)
        loss_value = loss_func(output, y_, cost, reg)
        if loss_value.requires_grad:
            loss_value.backward()
        return loss_value

    opt = optim.LBFGS(model.parameters(), max_iter=20, tolerance_grad=1e-07,
                      tolerance_change=1e-09, history_size=100)
    X_ = Variable(X, requires_grad=True)
    y_ = Variable(y)
    opt.step(closure)
    output, reg = model(X_)
    if classify:
        return closure(), get_accuracy(output, y)
    return closure()


def train(model, X, y, optimizer, device, loss_func=None, cost=.1):
    model.train()
    X = X.to(device)
    y = y.to(device)
    pred, reg = model(X)
    loss_value = loss_func(pred, y, cost, reg)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    return loss_value


def test(model, X, y, device, loss_func=None, cost=.1, classify=False):
    model.eval()

    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        pred, reg = model(X)
        loss_value = loss_func(pred, y, cost, reg)
        if classify:
            return loss_value, pred, get_accuracy(pred, y)

    return loss_value, pred


def get_accuracy(preds, y):
    output = preds.argmax(dim=1, keepdim=True)
    correct = output.eq(y.view_as(output)).sum().item()
    n_total = y.shape[0]
    return correct/n_total * 100
