"""Privacy-free FedAvg baseline for CIFAR-100 with ResNet18.

This entry point intentionally bypasses all DP/PFA/privacy-accountant code.
It follows the paper-style PF baseline setup after replacing CIFAR-10 with
CIFAR-100: ResNet18, Dirichlet non-IID partitioning, client sampling, and
vanilla FedAvg aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


CIFAR100_NUM_CLASSES = 100
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

PAPER_DEFAULT_NUM_CLIENTS = 20
PAPER_DEFAULT_CLIENT_FRACTION = 0.8
PAPER_DEFAULT_DIRICHLET_ALPHA = 0.3
PAPER_DEFAULT_GLOBAL_ROUNDS = 100
PAPER_DEFAULT_LOCAL_EPOCHS = 1
PAPER_DEFAULT_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-100 + ResNet18 + privacy-free FedAvg baseline."
    )
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--model", default="resnet18", choices=["resnet18"])
    parser.add_argument("--method", default="fedavg", choices=["fedavg"])
    parser.add_argument(
        "--no_dp",
        action="store_true",
        default=True,
        help="Accepted for explicit reproducibility; DP is always disabled here.",
    )

    parser.add_argument("--data_dir", default="./data")
    parser.add_argument(
        "--num_clients",
        type=int,
        default=PAPER_DEFAULT_NUM_CLIENTS,
        help=(
            "Total FL clients. Paper reports K in {20, 30, 40, 50}; "
            "default is K=20."
        ),
    )
    parser.add_argument(
        "--client_fraction",
        type=float,
        default=PAPER_DEFAULT_CLIENT_FRACTION,
        help="Client sample rate per round. Paper default is 0.8.",
    )
    parser.add_argument(
        "--partition",
        default="dirichlet",
        choices=["iid", "non-iid", "dirichlet"],
        help="Client data partition strategy.",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=PAPER_DEFAULT_DIRICHLET_ALPHA,
        help="Dirichlet scaling/concentration parameter. Paper CIFAR10 setting is 0.3.",
    )
    parser.add_argument(
        "--global_rounds",
        type=int,
        default=PAPER_DEFAULT_GLOBAL_ROUNDS,
        help="Global communication rounds. Paper default is T=100.",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=PAPER_DEFAULT_LOCAL_EPOCHS,
        help=(
            "Number of full local epochs when --local_update_mode=full-epoch. "
            "In random-batch mode, each selected client always updates on one "
            "random mini-batch per communication round."
        ),
    )
    parser.add_argument(
        "--local_update_mode",
        default="random-batch",
        choices=["random-batch", "full-epoch"],
        help=(
            "random-batch matches the paper-style local update: one shuffled "
            "mini-batch per selected client per communication round. "
            "full-epoch preserves the old behavior."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=PAPER_DEFAULT_BATCH_SIZE,
        help="Local random batch size. Paper default is B=16.",
    )
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument(
        "--limit_train_samples",
        type=int,
        default=None,
        help="Optional smoke-test limit before partitioning.",
    )
    parser.add_argument(
        "--limit_test_samples",
        type=int,
        default=None,
        help="Optional smoke-test limit for evaluation.",
    )
    parser.add_argument(
        "--output_csv",
        default="results/pf_fedavg_cifar100_resnet18_noniid_alpha0.3.csv",
        help="Where to save per-round train loss and test accuracy.",
    )
    parser.add_argument(
        "--run_config_json",
        default=None,
        help="Optional path for saving run configuration as JSON.",
    )
    parser.add_argument(
        "--run_config_csv",
        default=None,
        help="Where to save run configuration and final summary as CSV.",
    )
    parser.add_argument(
        "--client_distribution_csv",
        default=None,
        help="Where to save per-client CIFAR-100 label counts as CSV.",
    )
    parser.add_argument(
        "--client_distribution_json",
        default=None,
        help="Optional path for saving per-client CIFAR-100 label counts as JSON.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_cifar100_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return train_transform, test_transform


def build_resnet18_cifar100(num_classes: int = CIFAR100_NUM_CLASSES) -> nn.Module:
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model_fn(model_name: str) -> Callable[[], nn.Module]:
    if model_name == "resnet18":
        return build_resnet18_cifar100
    raise ValueError(f"Unsupported model: {model_name}")


def maybe_limit_dataset(dataset: Dataset, limit: Optional[int], seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def load_cifar100(
    data_dir: str,
    seed: int,
    limit_train: Optional[int],
    limit_test: Optional[int],
) -> Tuple[Dataset, Dataset]:
    train_transform, test_transform = build_cifar100_transforms()
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    train_dataset = maybe_limit_dataset(train_dataset, limit_train, seed)
    test_dataset = maybe_limit_dataset(test_dataset, limit_test, seed)
    return train_dataset, test_dataset


def iid_partition(dataset: Dataset, num_clients: int, seed: int) -> List[Subset]:
    if num_clients <= 0:
        raise ValueError("--num_clients must be positive.")
    if num_clients > len(dataset):
        raise ValueError("--num_clients cannot exceed the number of train samples.")

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(dataset), generator=generator).tolist()

    base_size = len(dataset) // num_clients
    remainder = len(dataset) % num_clients
    partitions = []
    cursor = 0
    for client_id in range(num_clients):
        client_size = base_size + (1 if client_id < remainder else 0)
        client_indices = shuffled_indices[cursor : cursor + client_size]
        partitions.append(Subset(dataset, client_indices))
        cursor += client_size
    return partitions


def get_dataset_targets(dataset: Dataset) -> List[int]:
    if isinstance(dataset, Subset):
        parent_targets = get_dataset_targets(dataset.dataset)
        return [int(parent_targets[int(index)]) for index in dataset.indices]
    if hasattr(dataset, "targets"):
        return [int(target) for target in dataset.targets]
    if hasattr(dataset, "labels"):
        return [int(target) for target in dataset.labels]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def ensure_min_client_samples(
    client_indices: List[List[int]],
    min_samples_per_client: int,
    rng: np.random.Generator,
) -> None:
    for client_id in range(len(client_indices)):
        while len(client_indices[client_id]) < min_samples_per_client:
            donor_id = max(range(len(client_indices)), key=lambda idx: len(client_indices[idx]))
            if len(client_indices[donor_id]) <= min_samples_per_client:
                raise RuntimeError(
                    "Unable to rebalance non-iid partition without creating an empty client."
                )
            donor_position = int(rng.integers(len(client_indices[donor_id])))
            client_indices[client_id].append(client_indices[donor_id].pop(donor_position))


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
) -> List[Subset]:
    if num_clients <= 0:
        raise ValueError("--num_clients must be positive.")
    if num_clients > len(dataset):
        raise ValueError("--num_clients cannot exceed the number of train samples.")
    if alpha <= 0:
        raise ValueError("--dirichlet_alpha must be positive.")

    rng = np.random.default_rng(seed)
    targets = np.asarray(get_dataset_targets(dataset), dtype=np.int64)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for class_id in sorted(np.unique(targets).tolist()):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        class_splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(int(index) for index in split.tolist())

    ensure_min_client_samples(client_indices, min_samples_per_client, rng)
    for indices in client_indices:
        rng.shuffle(indices)

    return [Subset(dataset, indices) for indices in client_indices]


def build_client_partitions(
    dataset: Dataset,
    num_clients: int,
    partition: str,
    dirichlet_alpha: float,
    seed: int,
) -> List[Subset]:
    if partition == "iid":
        return iid_partition(dataset, num_clients, seed)
    if partition in {"dirichlet", "non-iid"}:
        return dirichlet_partition(dataset, num_clients, dirichlet_alpha, seed)
    raise ValueError(f"Unsupported partition: {partition}")


def compute_client_label_distribution(
    client_datasets: Sequence[Dataset],
    num_classes: int,
) -> List[Dict[str, object]]:
    distribution = []
    for client_id, dataset in enumerate(client_datasets):
        targets = get_dataset_targets(dataset)
        class_counts = Counter(targets)
        label_counts = {
            str(class_id): int(class_counts.get(class_id, 0))
            for class_id in range(num_classes)
        }
        distribution.append(
            {
                "client_id": client_id,
                "total_samples": len(targets),
                "num_classes": sum(1 for count in label_counts.values() if count > 0),
                "label_counts": label_counts,
            }
        )
    return distribution


def log_client_label_distribution(distribution: Sequence[Dict[str, object]]) -> None:
    print("Client label distribution summary:")
    for item in distribution:
        print(
            f"  client {int(item['client_id']):02d}: "
            f"samples={int(item['total_samples'])}, "
            f"classes={int(item['num_classes'])}"
        )


def save_client_label_distribution_csv(
    path: str,
    distribution: Sequence[Dict[str, object]],
    num_classes: int,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as csvfile:
        fieldnames = ["client_id", "total_samples", "num_classes"] + [
            f"class_{class_id}" for class_id in range(num_classes)
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in distribution:
            label_counts = item["label_counts"]
            row = {
                "client_id": item["client_id"],
                "total_samples": item["total_samples"],
                "num_classes": item["num_classes"],
            }
            row.update(
                {
                    f"class_{class_id}": label_counts[str(class_id)]
                    for class_id in range(num_classes)
                }
            )
            writer.writerow(row)


def save_client_label_distribution_json(
    path: str,
    distribution: Sequence[Dict[str, object]],
    partition: str,
    dirichlet_alpha: float,
    num_classes: int,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "partition": partition,
        "dirichlet_alpha": dirichlet_alpha,
        "num_clients": len(distribution),
        "num_classes": num_classes,
        "clients": distribution,
    }
    with open(path, "w") as jsonfile:
        json.dump(payload, jsonfile, indent=2)


def resolve_run_config_csv_path(output_csv: str, run_config_csv: Optional[str]) -> str:
    if run_config_csv:
        return run_config_csv
    base_path, _ = os.path.splitext(output_csv)
    return f"{base_path}_config.csv"


def resolve_client_distribution_csv_path(
    output_csv: str,
    client_distribution_csv: Optional[str],
) -> str:
    if client_distribution_csv:
        return client_distribution_csv
    base_path, _ = os.path.splitext(output_csv)
    return f"{base_path}_client_distribution.csv"


def save_run_config_json(
    path: str,
    args: argparse.Namespace,
    client_distribution: Sequence[Dict[str, object]],
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        "baseline": "PF / PrivacyFree / vanilla FedAvg without DP",
        "paper_migration": {
            "source_pdf": "TNSE.pdf",
            "source_dataset": "CIFAR10",
            "target_dataset": "CIFAR100",
            "model": "ResNet18",
            "partition": "Dirichlet non-IID",
            "paper_dirichlet_alpha": PAPER_DEFAULT_DIRICHLET_ALPHA,
            "paper_client_fraction": PAPER_DEFAULT_CLIENT_FRACTION,
            "paper_global_rounds": PAPER_DEFAULT_GLOBAL_ROUNDS,
            "paper_batch_size": PAPER_DEFAULT_BATCH_SIZE,
            "paper_style_local_update": "one random mini-batch per selected client per round",
            "dp_enabled": False,
        },
        "args": vars(args),
        "effective_local_updates_per_round": (
            1 if args.local_update_mode == "random-batch" else args.local_epochs
        ),
        "client_summary": [
            {
                "client_id": item["client_id"],
                "total_samples": item["total_samples"],
                "num_classes": item["num_classes"],
            }
            for item in client_distribution
        ],
    }
    with open(path, "w") as jsonfile:
        json.dump(payload, jsonfile, indent=2)


def save_run_config_csv(
    path: str,
    args: argparse.Namespace,
    client_distribution: Sequence[Dict[str, object]],
    final_test_accuracy: Optional[float] = None,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    total_samples = sum(int(item["total_samples"]) for item in client_distribution)
    min_client_samples = min(int(item["total_samples"]) for item in client_distribution)
    max_client_samples = max(int(item["total_samples"]) for item in client_distribution)
    min_client_classes = min(int(item["num_classes"]) for item in client_distribution)
    max_client_classes = max(int(item["num_classes"]) for item in client_distribution)

    rows = [
        ("baseline", "name", "PF / PrivacyFree / vanilla FedAvg without DP"),
        ("paper_migration", "source_pdf", "TNSE.pdf"),
        ("paper_migration", "source_dataset", "CIFAR10"),
        ("paper_migration", "target_dataset", "CIFAR100"),
        ("paper_migration", "model", "ResNet18"),
        ("paper_migration", "partition", "Dirichlet non-IID"),
        ("paper_migration", "paper_dirichlet_alpha", PAPER_DEFAULT_DIRICHLET_ALPHA),
        ("paper_migration", "paper_client_fraction", PAPER_DEFAULT_CLIENT_FRACTION),
        ("paper_migration", "paper_global_rounds", PAPER_DEFAULT_GLOBAL_ROUNDS),
        ("paper_migration", "paper_batch_size", PAPER_DEFAULT_BATCH_SIZE),
        (
            "paper_migration",
            "paper_style_local_update",
            "one random mini-batch per selected client per round",
        ),
        ("paper_migration", "dp_enabled", False),
        ("client_summary", "total_samples", total_samples),
        ("client_summary", "min_client_samples", min_client_samples),
        ("client_summary", "max_client_samples", max_client_samples),
        ("client_summary", "min_client_classes", min_client_classes),
        ("client_summary", "max_client_classes", max_client_classes),
    ]

    for key, value in sorted(vars(args).items()):
        rows.append(("args", key, value))

    effective_local_updates = (
        1 if args.local_update_mode == "random-batch" else args.local_epochs
    )
    rows.append(("effective", "local_updates_per_round", effective_local_updates))

    if final_test_accuracy is not None:
        rows.append(("result", "final_test_accuracy", f"{final_test_accuracy:.6f}"))

    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["section", "key", "value"])
        writer.writerows(rows)


def make_train_loaders(
    client_datasets: Sequence[Dataset],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> List[DataLoader]:
    loaders = []
    for client_id, dataset in enumerate(client_datasets):
        generator = torch.Generator().manual_seed(seed + client_id)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                generator=generator,
            )
        )
    return loaders


def make_test_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def select_clients(
    num_clients: int,
    client_fraction: float,
    round_idx: int,
    seed: int,
) -> List[int]:
    if not 0 < client_fraction <= 1:
        raise ValueError("--client_fraction must be in (0, 1].")
    num_selected = max(1, int(math.ceil(num_clients * client_fraction)))
    generator = torch.Generator().manual_seed(seed + round_idx)
    selected = torch.randperm(num_clients, generator=generator)[:num_selected]
    return sorted(selected.tolist())


def clone_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in state_dict.items())


def train_minibatch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> Tuple[float, int]:
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    batch_size = targets.size(0)
    return loss.item() * batch_size, batch_size


def train_client(
    model_fn: Callable[[], nn.Module],
    global_state: OrderedDict[str, torch.Tensor],
    train_loader: DataLoader,
    local_epochs: int,
    local_update_mode: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
) -> Tuple[OrderedDict[str, torch.Tensor], float, int]:
    model = model_fn().to(device)
    model.load_state_dict(global_state)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    total_loss = 0.0
    total_examples = 0
    if local_update_mode == "random-batch":
        inputs, targets = next(iter(train_loader))
        batch_loss, batch_size = train_minibatch(
            model,
            criterion,
            optimizer,
            inputs,
            targets,
            device,
        )
        total_loss += batch_loss
        total_examples += batch_size
    elif local_update_mode == "full-epoch":
        if local_epochs <= 0:
            raise ValueError("--local_epochs must be positive.")
        for _ in range(local_epochs):
            for inputs, targets in train_loader:
                batch_loss, batch_size = train_minibatch(
                    model,
                    criterion,
                    optimizer,
                    inputs,
                    targets,
                    device,
                )
                total_loss += batch_loss
                total_examples += batch_size
    else:
        raise ValueError(f"Unsupported local update mode: {local_update_mode}")

    avg_loss = total_loss / max(1, total_examples)
    return clone_state_dict(model.state_dict()), avg_loss, len(train_loader.dataset)


def fedavg_aggregate(
    client_states: Sequence[OrderedDict[str, torch.Tensor]],
    client_sizes: Sequence[int],
) -> OrderedDict[str, torch.Tensor]:
    if not client_states:
        raise ValueError("Cannot aggregate an empty client state list.")

    total_size = float(sum(client_sizes))
    aggregated = OrderedDict()
    for name in client_states[0].keys():
        first_value = client_states[0][name]
        if torch.is_floating_point(first_value):
            value = torch.zeros_like(first_value)
            for state, size in zip(client_states, client_sizes):
                value += state[name] * (size / total_size)
            aggregated[name] = value
        else:
            aggregated[name] = first_value.clone()
    return aggregated


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def init_output_csv(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "round",
                "selected_clients",
                "train_loss",
                "test_loss",
                "test_accuracy",
            ]
        )


def append_output_csv(
    path: str,
    round_idx: int,
    selected_clients: Iterable[int],
    train_loss: float,
    test_loss: float,
    test_accuracy: float,
) -> None:
    with open(path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                round_idx,
                " ".join(str(client_id) for client_id in selected_clients),
                f"{train_loss:.6f}",
                f"{test_loss:.6f}",
                f"{test_accuracy:.6f}",
            ]
        )


def validate_model_output(model_fn: Callable[[], nn.Module]) -> None:
    model = model_fn()
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 32, 32))
    expected_shape = (2, 100)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Expected model output shape {expected_shape}, got {tuple(output.shape)}."
        )


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    model_fn = build_model_fn(args.model)
    validate_model_output(model_fn)

    train_dataset, test_dataset = load_cifar100(
        data_dir=args.data_dir,
        seed=args.seed,
        limit_train=args.limit_train_samples,
        limit_test=args.limit_test_samples,
    )
    client_datasets = build_client_partitions(
        train_dataset,
        num_clients=args.num_clients,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
    )
    client_label_distribution = compute_client_label_distribution(
        client_datasets,
        num_classes=CIFAR100_NUM_CLASSES,
    )
    args.run_config_csv = resolve_run_config_csv_path(
        args.output_csv,
        args.run_config_csv,
    )
    args.client_distribution_csv = resolve_client_distribution_csv_path(
        args.output_csv,
        args.client_distribution_csv,
    )
    save_client_label_distribution_csv(
        args.client_distribution_csv,
        client_label_distribution,
        num_classes=CIFAR100_NUM_CLASSES,
    )
    if args.client_distribution_json:
        save_client_label_distribution_json(
            args.client_distribution_json,
            client_label_distribution,
            partition=args.partition,
            dirichlet_alpha=args.dirichlet_alpha,
            num_classes=CIFAR100_NUM_CLASSES,
        )
    if args.run_config_json:
        save_run_config_json(
            args.run_config_json,
            args,
            client_label_distribution,
        )
    save_run_config_csv(
        args.run_config_csv,
        args,
        client_label_distribution,
    )
    train_loaders = make_train_loaders(
        client_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = make_test_loader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    global_model = model_fn().to(device)
    global_state = clone_state_dict(global_model.state_dict())
    init_output_csv(args.output_csv)

    print("Baseline: CIFAR-100 + ResNet18 + PF/FedAvg + no-DP")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}, test samples: {len(test_dataset)}")
    print(f"Clients: {args.num_clients}, client_fraction: {args.client_fraction}")
    print(f"Global rounds: {args.global_rounds}")
    print(f"Local update mode: {args.local_update_mode}")
    if args.local_update_mode == "random-batch":
        print("Local update: one random mini-batch per selected client per round")
    else:
        print(f"Local update: {args.local_epochs} full local epochs per selected client")
    print(f"Batch size: {args.batch_size}, test_batch_size: {args.test_batch_size}")
    print(
        f"Learning rate: {args.lr}, momentum: {args.momentum}, "
        f"weight_decay: {args.weight_decay}"
    )
    print(f"Partition: {args.partition}")
    print(f"Dirichlet alpha: {args.dirichlet_alpha}")
    log_client_label_distribution(client_label_distribution)
    print(f"Client distribution CSV saved to: {args.client_distribution_csv}")
    if args.client_distribution_json:
        print(f"Client distribution JSON saved to: {args.client_distribution_json}")
    if args.run_config_json:
        print(f"Run config JSON saved to: {args.run_config_json}")
    print(f"Run config CSV saved to: {args.run_config_csv}")
    print("DP, clipping, noise, epsilon, delta, and accountants are disabled.")

    final_test_acc = 0.0
    for round_idx in range(1, args.global_rounds + 1):
        selected_clients = select_clients(
            args.num_clients, args.client_fraction, round_idx, args.seed
        )

        client_states = []
        client_sizes = []
        weighted_loss_sum = 0.0
        for client_id in selected_clients:
            client_state, client_loss, client_size = train_client(
                model_fn=model_fn,
                global_state=global_state,
                train_loader=train_loaders[client_id],
                local_epochs=args.local_epochs,
                local_update_mode=args.local_update_mode,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                device=device,
            )
            client_states.append(client_state)
            client_sizes.append(client_size)
            weighted_loss_sum += client_loss * client_size

        train_loss = weighted_loss_sum / float(sum(client_sizes))
        global_state = fedavg_aggregate(client_states, client_sizes)
        global_model.load_state_dict(global_state)

        if round_idx % args.eval_every == 0 or round_idx == args.global_rounds:
            test_loss, test_accuracy = evaluate(global_model, test_loader, device)
            final_test_acc = test_accuracy
        else:
            test_loss, test_accuracy = float("nan"), float("nan")

        append_output_csv(
            args.output_csv,
            round_idx=round_idx,
            selected_clients=selected_clients,
            train_loss=train_loss,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
        )
        print(
            f"Round {round_idx:03d}/{args.global_rounds} | "
            f"clients={selected_clients} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_accuracy:.4f}"
        )

    print(f"Final test accuracy: {final_test_acc:.4f}")
    print(f"Metrics saved to: {args.output_csv}")
    save_run_config_csv(
        args.run_config_csv,
        args,
        client_label_distribution,
        final_test_accuracy=final_test_acc,
    )
    print(f"Run summary CSV saved to: {args.run_config_csv}")


if __name__ == "__main__":
    main()
