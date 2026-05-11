"""图像 + 文本多模态分类模型的训练入口。

本脚本负责构建共享词表的数据加载器、初始化 MultimodalModel、
执行训练/验证循环，并保存训练日志和最佳模型权重。
"""

import csv
import os
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    LOG_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    RANDOM_SEED,
    TRAIN_IMAGE_DIR,
    TRAIN_METADATA,
    VAL_IMAGE_DIR,
    VAL_METADATA,
    ensure_output_dirs,
)
from dataset import FoodDataset
from models import MultimodalModel


def create_dataloaders() -> Dict[str, DataLoader]:
    """创建训练集和验证集 DataLoader，并让二者共享训练集词表。"""
    train_dataset = FoodDataset(
        metadata_path=TRAIN_METADATA,
        image_dir=TRAIN_IMAGE_DIR,
        train=True,
    )
    val_dataset = FoodDataset(
        metadata_path=VAL_METADATA,
        image_dir=VAL_IMAGE_DIR,
        vocab=train_dataset.vocab,
        train=False,
    )

    pin_memory = torch.cuda.is_available()
    # Windows 下多进程 DataLoader 容易受权限和启动方式影响，这里默认使用单进程加载。
    num_workers = 0 if os.name == "nt" else NUM_WORKERS
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """训练一个 epoch，并返回平均 loss 和 accuracy。"""
    model.train()

    running_loss = 0.0
    running_correct = 0
    sample_count = 0

    for batch in dataloader:
        images = batch["image_tensor"].to(device)
        text_ids = batch["text_ids"].to(device)
        labels = batch["label_id"].to(device)

        # 多模态模型同时输入图片张量和文本 token id。
        optimizer.zero_grad()
        outputs = model(images, text_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        sample_count += batch_size

    return {
        "loss": running_loss / sample_count,
        "acc": running_correct / sample_count,
    }


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """验证一个 epoch，并返回平均 loss 和 accuracy。"""
    model.eval()

    running_loss = 0.0
    running_correct = 0
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image_tensor"].to(device)
            text_ids = batch["text_ids"].to(device)
            labels = batch["label_id"].to(device)

            outputs = model(images, text_ids)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            predictions = outputs.argmax(dim=1)
            running_loss += loss.item() * batch_size
            running_correct += (predictions == labels).sum().item()
            sample_count += batch_size

    return {
        "loss": running_loss / sample_count,
        "acc": running_correct / sample_count,
    }


def append_log_row(log_path: Path, row: Dict[str, float]) -> None:
    """向 CSV 日志文件追加一行训练结果。"""
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    file_exists = log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """在训练集上训练 MultimodalModel，并在验证集上评估。"""
    ensure_output_dirs()

    # 将预训练权重下载缓存放在项目 outputs 目录下。
    torch_cache_dir = OUTPUT_DIR / "torch_cache"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache_dir))

    # 固定随机种子，便于后续复现实验。
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = create_dataloaders()
    vocab_size = len(dataloaders["train"].dataset.vocab)

    model = MultimodalModel(
        vocab_size=vocab_size,
        num_classes=NUM_CLASSES,
        pretrained=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    log_path = LOG_DIR / "multimodal_train_log.csv"
    checkpoint_path = CHECKPOINT_DIR / "multimodal_best.pth"
    if log_path.exists():
        log_path.unlink()

    best_val_acc = float("-inf")

    print(f"Device: {device}")
    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    print(f"Val samples: {len(dataloaders['val'].dataset)}")
    print(f"Vocab size: {vocab_size}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )

        log_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        append_log_row(log_path, log_row)

        # 多模态模型需要同时保存词表，测试和推理时必须使用同一套 token 映射。
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_metrics["acc"],
                    "val_loss": val_metrics["loss"],
                    "vocab": dataloaders["train"].dataset.vocab,
                    "vocab_size": vocab_size,
                    "num_classes": NUM_CLASSES,
                },
                checkpoint_path,
            )

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f}"
        )

    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Log saved to: {log_path}")
    print(f"Best checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
