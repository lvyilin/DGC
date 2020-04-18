import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from .CelebA import CelebA

_torch_supported_dataset = ('mnist',)
_custom_dataset = {'celeba'}
_torch_dataset_key_mapping = {
    'mnist': 'MNIST',
    'celeba': 'CelebA',
}
_dataset_ratio_mapping = {
    'mnist': [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40],
    'celeba': [15000, 1500, 750, 300, 150]
}


def dataset_to_numpy(dataset):
    loader = DataLoader(dataset, len(dataset))
    x, y = next(iter(loader))
    return x.numpy(), y.numpy()


def load_data(name, seed, imbalance=None, data_dir=None):
    name = name.lower()
    if data_dir is None:
        data_dir = './data/%s/' % name

    if name in _torch_supported_dataset:
        func_name = _torch_dataset_key_mapping[name]

        dataset_func = getattr(torchvision.datasets, func_name)
        transform = transforms.Compose([transforms.ToTensor(), ])
        if name == 'mnist':
            train_dataset = dataset_func(data_dir, train=True, transform=transform, download=True)
            test_dataset = dataset_func(data_dir, train=False, transform=transform, download=True)
        else:
            raise NotImplementedError
    elif name in _custom_dataset:
        if name == 'celeba':
            train_dataset = CelebA(data_dir, split='train', download=True)
            test_dataset = CelebA(data_dir, split='test', download=True)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)
    X_train, y_train = _shuffle(X_train, y_train, seed)
    X_train = np.transpose(X_train, axes=[0, 2, 3, 1])
    X_test = np.transpose(X_test, axes=[0, 2, 3, 1])
    n_classes = len(np.unique(y_test))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print([np.count_nonzero(y_train == i) for i in range(n_classes)])
    if imbalance is None or imbalance is False:
        return (X_train, y_train), (X_test, y_test)
    if imbalance is True:
        ratio = _dataset_ratio_mapping[name]
    else:
        ratio = imbalance
    X_train = [X_train[y_train == i][:num] for i, num in enumerate(ratio)]
    y_train = [y_train[y_train == i][:num] for i, num in enumerate(ratio)]
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_train, y_train = _shuffle(X_train, y_train, seed)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return (X_train, y_train), (X_test, y_test)


def _shuffle(x, y, seed):
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    return x, y
