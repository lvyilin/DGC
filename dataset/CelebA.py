import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torchvision
from torchvision.transforms import transforms


class CelebA(torchvision.datasets.CelebA):
    training_file = 'training.pt'
    test_file = 'test.pt'
    valid_file = 'valid.pt'
    target_size = 64

    def __init__(self, root, split,
                 columns=('Blond_Hair', 'Black_Hair', 'Bald', 'Brown_Hair', 'Gray_Hair'),
                 download=False):
        super().__init__(root, split='all', target_type="attr", download=download)
        preprocessed = self.has_preprocessed()
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
        }
        assert split in split_map
        split = split_map[split]
        splits = pandas.read_csv(os.path.join(self.root, self.base_folder, "list_eval_partition.txt"),
                                 delim_whitespace=True, header=None, index_col=0)[1].values
        if not preprocessed:
            print('Processing...')
            img_folder = os.path.join(self.root, self.base_folder, "img_align_celeba")
            # images, labels = self.process_data(img_folder, columns)
            column_idx = [self.attr_names.index(col) for col in columns]
            data = []
            split_list = []
            for i, c in enumerate(column_idx):
                attr_c = self.attr[:, c] == 1
                split_list.append(splits[attr_c])
                filenames_c = self.filename[attr_c.numpy()]
                for filename in filenames_c:
                    image = plt.imread(os.path.join(img_folder, filename))
                    image = self.resize(image)
                    label = i
                    data.append((image, label))
            images = torch.stack([i[0] for i in data], dim=0)
            labels = torch.tensor([i[1] for i in data], dtype=torch.long)
            splits = np.concatenate(split_list, axis=0)

            for i, filename in ((0, self.training_file),
                                (1, self.valid_file),
                                (2, self.test_file)):
                mask = (splits == i)
                self.save_data(os.path.join(self.root, filename),
                               (images[mask], labels[mask]))
            print('Done!')

        if split == 0:
            data_file = self.training_file
        elif split == 1:
            data_file = self.valid_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.root, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def __len__(self):
        return len(self.data)

    def has_preprocessed(self):
        return os.path.isfile(os.path.join(self.root, self.training_file)) and \
               os.path.isfile(os.path.join(self.root, self.valid_file)) and \
               os.path.isfile(os.path.join(self.root, self.test_file))

    def save_data(self, file, data):
        with open(file, 'wb') as f:
            torch.save(data, f)

    def resize(self, img):
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
        ])
        return trans(img)
