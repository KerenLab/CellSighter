import sys
sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import argparse
import numpy as np
from model import Model
from data.data import CellCropsDataset
from data.utils import load_crops
from data.transform import train_transform, val_transform
from torch.utils.data import DataLoader, WeightedRandomSampler
import json
from metrics.metrics import Metrics
from eval import val_epoch


def train_epoch(model, dataloader, optimizer, criterion, epoch, writer, device=None):
    model.train()
    cells = []
    for i, batch in enumerate(dataloader):
        x = batch['image']
        m = batch.get('mask', None)
        if m is not None:
            x = torch.cat([x, m], dim=1)
        x = x.to(device=device)

        y = batch['label'].to(device=device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if i % 100 == 0:
            print(f"epoch {epoch} | iterate {i} / {len(dataloader)} | {loss.item()}")
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)
        loss.backward()
        optimizer.step()
    return cells


def subsample_const_size(crops, size):
    """
    sample same number of cell from each class
    """
    final_crops = []
    crops = np.array(crops)
    labels = np.array([c._label for c in crops])
    for lbl in np.unique(labels):
        indices = np.argwhere(labels == lbl).flatten()
        if (labels == lbl).sum() < size:
            chosen_indices = indices
        else:
            chosen_indices = np.random.choice(indices, size, replace=False)
        final_crops += crops[chosen_indices].tolist()
    return final_crops


def define_sampler(crops, hierarchy_match=None):
    """
    Sampler that sample from each cell category equally
    The hierarchy_match defines the cell category for each class.
    if None then each class will be category of it's own.
    """
    labels = np.array([c._label for c in crops])
    if hierarchy_match is not None:
        labels = np.array([hierarchy_match[str(l)] for l in labels])

    unique_labels = np.unique(labels)
    class_sample_count = {t: len(np.where(labels == t)[0]) for t in unique_labels}
    weight = {k: sum(class_sample_count.values()) / v for k, v in class_sample_count.items()}
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)
    return WeightedRandomSampler(samples_weight.double(), len(samples_weight))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--base_path', type=str,
                        help='configuration_path')
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.base_path)
    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    criterion = torch.nn.CrossEntropyLoss()
    train_crops, val_crops = load_crops(config["root_dir"],
                                        config["channels_path"],
                                        config["crop_size"],
                                        config["train_set"],
                                        config["val_set"],
                                        config["to_pad"],
                                        blacklist_channels=config["blacklist"])

    train_crops = np.array([c for c in train_crops if c._label >= 0])
    val_crops = np.array([c for c in val_crops if c._label >= 0])
    if "size_data" in config:
        train_crops = subsample_const_size(train_crops, config["size_data"])
    sampler = define_sampler(train_crops, config["hierarchy_match"])
    shift = 5
    crop_input_size = config["crop_input_size"] if "crop_input_size" in config else 100
    aug = config["aug"] if "aug" in config else True
    training_transform = train_transform(crop_input_size, shift) if aug else val_transform(crop_input_size)
    train_dataset = CellCropsDataset(train_crops, transform=training_transform, mask=True)
    val_dataset = CellCropsDataset(val_crops, transform=val_transform(crop_input_size), mask=True)
    train_dataset_for_eval = CellCropsDataset(train_crops, transform=val_transform(crop_input_size), mask=True)
    device = "cuda"
    num_channels = sum(1 for line in open(config["channels_path"])) + 1 - len(config["blacklist"])
    class_num = config["num_classes"]

    model = Model(num_channels + 1, class_num)

    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    train_loader = DataLoader(train_dataset, batch_size=128,
                              num_workers=8,
                              sampler=sampler if config["sample_batch"] else None,
                              shuffle=False if config["sample_batch"] else True)
    train_loader_for_eval = DataLoader(train_dataset_for_eval, batch_size=128,
                                       num_workers=8, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128,
                            num_workers=10, shuffle=False)
    print(len(train_loader), len(val_loader))
    for i in range(config["epoch_max"]):
        train_epoch(model, train_loader, optimizer, criterion, device=device, epoch=i, writer=writer)
        print(f"Epoch {i} done!")
        torch.save(model.state_dict(), os.path.join(args.base_path, f"./weights_{i}_count.pth"))
        if (i % 10 == 0) & (i > 0):
            cells_val, results_val = val_epoch(model, val_loader, device=device)
            metrics = Metrics([],
                              writer,
                              prefix="val")
            metrics(cells_val, results_val, i)
            metrics.save_results(os.path.join(args.base_path, f"val_results_{i}.csv"), cells_val, results_val)
            #  TODO uncooment to eval on the train as well
            # cells_train, results_train = val_epoch(model, train_loader_for_eval, device=device)
            #  metrics = Metrics(
            #     [],
            #     writer,
            #     prefix="train")
            # metrics(cells_train, results_train, i)
            # metrics.save_results(os.path.join(args.base_path, f"train_results_{i}.csv"), cells_train, results_train)
