# https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn


class KNNValidation:
    def __init__(self, model, K=1):
        self.model = model
        self.K = K

        base_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root="/data/datasets", train=True, download=True, transform=base_transforms
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        val_dataset = datasets.CIFAR10(
            root="/data/datasets", train=False, download=True, transform=base_transforms
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def eval(self, device):
        """Extract features from validation split and search on train split features."""
        n_data = self.train_dataloader.dataset.data.shape[0]
        feat_dim = 512

        self.model.eval()
        torch.cuda.empty_cache()

        train_features = torch.zeros([feat_dim, n_data], device=device)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
                inputs = inputs.to(device)
                batch_size = inputs.size(0)

                # forward
                features = self.model(inputs)
                features = nn.functional.normalize(features)
                train_features[
                    :, batch_idx * batch_size : batch_idx * batch_size + batch_size
                ] = features.data.t()

            train_labels = torch.LongTensor(self.train_dataloader.dataset.targets).to(device)

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_dataloader):
                targets = targets.to(device, non_blocking=True)
                batch_size = inputs.size(0)
                features = self.model(inputs.to(device))

                dist = torch.mm(features, train_features)
                yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

                total += targets.size(0)
                correct += retrieval.eq(targets.data).sum().item()
        top1 = correct / total

        return top1
