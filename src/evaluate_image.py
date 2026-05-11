"""在测试集上评估训练好的仅图像分类模型。"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    LOG_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    TEST_IMAGE_DIR,
    TEST_METADATA,
    ensure_output_dirs,
)
from dataset import build_image_transform
from models import ImageOnlyModel


class ImageOnlyTestDataset(Dataset):
    """仅图像模型测试集，只返回图片张量和标签。"""

    def __init__(self, metadata_path: Path, image_dir: Path) -> None:
        self.metadata_path = Path(metadata_path)
        self.image_dir = Path(image_dir)
        self.transform = build_image_transform(train=False)
        self.data = pd.read_csv(self.metadata_path, usecols=["image_path", "label"])

        unknown_labels = sorted(set(self.data["label"]) - set(CLASS_TO_IDX))
        if unknown_labels:
            raise ValueError(f"{self.metadata_path} 中存在未知类别: {unknown_labels}")

    def __len__(self) -> int:
        """返回测试集样本数量。"""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """读取单个测试样本。"""
        row = self.data.iloc[index]
        image_path = self.image_dir / row["image_path"]
        label_id = CLASS_TO_IDX[row["label"]]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        return {
            "image_tensor": image_tensor,
            "label_id": torch.tensor(label_id, dtype=torch.long),
        }


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict:
    """从训练保存的 checkpoint 中加载模型权重。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def select_device() -> torch.device:
    """选择运行设备；仅在当前 PyTorch 构建支持 GPU 架构时使用 CUDA。"""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    capability = torch.cuda.get_device_capability()
    device_arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = torch.cuda.get_arch_list()
    if supported_arches and device_arch not in supported_arches:
        print(
            f"当前 CUDA 设备架构 {device_arch} 不被此 PyTorch 构建支持，"
            "将自动切换到 CPU。"
        )
        return torch.device("cpu")

    return torch.device("cuda")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """在测试集上评估模型，并返回 loss、accuracy、真实标签和预测标签。"""
    model.eval()

    running_loss = 0.0
    running_correct = 0
    sample_count = 0
    all_labels: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image_tensor"].to(device)
            labels = batch["label_id"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (predictions == labels).sum().item()
            sample_count += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return (
        running_loss / sample_count,
        running_correct / sample_count,
        all_labels,
        all_predictions,
    )


def build_metrics_text(
    test_loss: float,
    test_acc: float,
    labels: List[int],
    predictions: List[int],
    checkpoint_path: Path,
    device: torch.device,
    sample_count: int,
) -> str:
    """将测试指标整理成便于阅读和保存的文本报告。"""
    report = classification_report(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
    )

    lines = [
        "ImageOnlyModel 测试集评估",
        f"设备: {device}",
        f"Checkpoint: {checkpoint_path}",
        f"测试样本数: {sample_count}",
        f"test_loss: {test_loss:.6f}",
        f"test_acc: {test_acc:.6f}",
        "",
        "各类别 precision / recall / f1-score:",
        report,
        "混淆矩阵:",
        "行表示真实标签，列表示预测标签",
        "类别顺序: " + ", ".join(CLASS_NAMES),
        str(matrix),
    ]
    return "\n".join(lines)


def main() -> None:
    """加载仅图像模型的最佳 checkpoint，并在测试集上评估。"""
    ensure_output_dirs()

    torch_cache_dir = OUTPUT_DIR / "torch_cache"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache_dir))

    device = select_device()
    checkpoint_path = CHECKPOINT_DIR / "image_only_best.pth"
    metrics_path = LOG_DIR / "image_test_metrics.txt"

    test_dataset = ImageOnlyTestDataset(
        metadata_path=TEST_METADATA,
        image_dir=TEST_IMAGE_DIR,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ImageOnlyModel(num_classes=NUM_CLASSES, pretrained=False).to(device)
    load_checkpoint(model, checkpoint_path, device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, labels, predictions = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    metrics_text = build_metrics_text(
        test_loss=test_loss,
        test_acc=test_acc,
        labels=labels,
        predictions=predictions,
        checkpoint_path=checkpoint_path,
        device=device,
        sample_count=len(test_dataset),
    )
    metrics_path.write_text(metrics_text + "\n", encoding="utf-8")

    print(metrics_text)
    print(f"\n测试指标已保存到: {metrics_path}")


if __name__ == "__main__":
    main()
