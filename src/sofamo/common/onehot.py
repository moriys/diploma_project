# %%
import torch
import numpy as np


# %%
def make_onehot_tensor(labels, n_classes):
    s = labels.shape
    n = n_classes
    dim = len(s)

    labels = labels.unsqueeze(0)
    onehot = torch.zeros((n, *s), device=labels.device)
    onehot.scatter_(0, labels.long(), 1)

    if dim == 3:
        onehot.transpose_(0, 1)
    elif dim == 4 and s[1] == 1:
        onehot.transpose_(0, 1).squeeze_(2)

    return onehot


# %%
def make_onehot_numpy(labels, n_classes):
    s = labels.shape
    n = n_classes
    dim = len(s)

    # labels = labels.flatten()
    # labels = labels.ravel()
    labels = labels.reshape((-1,))
    onehot = np.zeros((n, len(labels)))
    for i in range(n):
        onehot[i, labels == i] = 1
    onehot = onehot.reshape(n, *s)

    if dim == 3:
        onehot = onehot.swapaxes(0, 1)
    elif dim == 4 and s[1] == 1:
        onehot = onehot.swapaxes(0, 1).squeeze(2)

    return onehot


# %%
def make_onehot(labels, n_classes):
    if isinstance(labels, (list, tuple)):
        labels = np.asarray(labels)
        func = make_onehot_numpy
    elif isinstance(labels, np.ndarray):
        func = make_onehot_numpy
    elif isinstance(labels, torch.Tensor):
        func = make_onehot_tensor
    onehot = func(labels, n_classes)
    return onehot


# %%
if __name__=='__main__':
    num_classes = 4
    torch.manual_seed(42)
    a = torch.randint(high=num_classes, size=(1,3,3))
    a = a.cpu()
    # a = a.cuda()
    # a = a.numpy()
    print(a)
    print(a.shape)
    b = make_onehot(a, num_classes)
    print(b)
    print(b.shape)

    import timeit
    import copy

    a = torch.randint(high=num_classes, size=(3,3))
    a_numpy = copy.deepcopy(a.numpy())
    a_tensor_cpu = copy.deepcopy(a.cpu())
    a_tensor_gpu = copy.deepcopy(a.cuda())

    def test_v0():
        b = make_onehot(a_numpy, num_classes)

    def test_v1():
        b = make_onehot(a_tensor_cpu, num_classes)

    def test_v2():
        b = make_onehot(a_tensor_gpu, num_classes)

    print(timeit.timeit(test_v0, number=100000))
    print(timeit.timeit(test_v1, number=100000))
    print(timeit.timeit(test_v2, number=100000))
