"""Pure FedAvg baseline for CIFAR-100 with a CIFAR-style ResNet18.

This entry point intentionally bypasses all DP/PFA/privacy-accountant code.
It is meant to provide a simple no-DP baseline for later comparisons.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-100 + ResNet18 + no-DP FedAvg baseline."
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
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--client_fraction", type=float, default=1.0)
    parser.add_argument("--global_rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument(
        "--limit_train_samples",
        type=int,
        default=None,
        help="Optional smoke-test limit before IID partitioning.",
    )
    parser.add_argument(
        "--limit_test_samples",
        type=int,
        default=None,
        help="Optional smoke-test limit for evaluation.",
    )
    parser.add_argument(
        "--output_csv",
        default="results/baseline_fedavg_cifar100_resnet18.csv",
        help="Where to save per-round train loss and test accuracy.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
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


def build_resnet18_cifar100(num_classes: int = 100) -> nn.Module:
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


def train_client(
    model_fn: Callable[[], nn.Module],
    global_state: OrderedDict[str, torch.Tensor],
    train_loader: DataLoader,
    local_epochs: int,
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
    for _ in range(local_epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

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
    client_datasets = iid_partition(train_dataset, args.num_clients, args.seed)
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

    print("Baseline: CIFAR-100 + ResNet18 + FedAvg + no-DP")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}, test samples: {len(test_dataset)}")
    print(f"Clients: {args.num_clients}, client_fraction: {args.client_fraction}")
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


if __name__ == "__main__":
    main()
