"""在测试集上评估训练好的图像 + 文本多模态分类模型。"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    LOG_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    TEST_IMAGE_DIR,
    TEST_METADATA,
    ensure_output_dirs,
)
from dataset import FoodDataset
from models import MultimodalModel


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


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict:
    """加载多模态模型 checkpoint 字典。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"期望 checkpoint 为字典，实际类型为 {type(checkpoint).__name__}")
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint 中缺少 'model_state_dict'。")
    if "vocab" not in checkpoint:
        raise KeyError("checkpoint 中缺少训练阶段词表 'vocab'。")

    return checkpoint


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
            text_ids = batch["text_ids"].to(device)
            labels = batch["label_id"].to(device)

            outputs = model(images, text_ids)
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


def compute_confusion_matrix(
    labels: List[int],
    predictions: List[int],
    num_classes: int,
) -> List[List[int]]:
    """计算混淆矩阵：行表示真实标签，列表示预测标签。"""
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, prediction in zip(labels, predictions):
        matrix[label][prediction] += 1
    return matrix


def build_classification_report(
    labels: List[int],
    predictions: List[int],
    matrix: List[List[int]],
) -> str:
    """不额外依赖 sklearn，手动生成类似 classification_report 的文本报告。"""
    lines = [
        f"{'':>14} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    ]

    total_support = len(labels)
    total_correct = sum(matrix[class_idx][class_idx] for class_idx in range(NUM_CLASSES))
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for class_idx, class_name in enumerate(CLASS_NAMES):
        true_positive = matrix[class_idx][class_idx]
        predicted_positive = sum(matrix[row_idx][class_idx] for row_idx in range(NUM_CLASSES))
        actual_positive = sum(matrix[class_idx])

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1_score
        weighted_precision += precision * actual_positive
        weighted_recall += recall * actual_positive
        weighted_f1 += f1_score * actual_positive

        lines.append(
            f"{class_name:>14} {precision:>10.4f} {recall:>10.4f} "
            f"{f1_score:>10.4f} {actual_positive:>10d}"
        )

    accuracy = total_correct / total_support if total_support else 0.0
    macro_precision /= NUM_CLASSES
    macro_recall /= NUM_CLASSES
    macro_f1 /= NUM_CLASSES
    weighted_precision /= total_support
    weighted_recall /= total_support
    weighted_f1 /= total_support

    lines.extend(
        [
            "",
            f"{'accuracy':>14} {'':>10} {'':>10} {accuracy:>10.4f} {total_support:>10d}",
            f"{'macro avg':>14} {macro_precision:>10.4f} {macro_recall:>10.4f} "
            f"{macro_f1:>10.4f} {total_support:>10d}",
            f"{'weighted avg':>14} {weighted_precision:>10.4f} {weighted_recall:>10.4f} "
            f"{weighted_f1:>10.4f} {total_support:>10d}",
        ]
    )
    return "\n".join(lines)


def build_metrics_text(
    test_loss: float,
    test_acc: float,
    labels: List[int],
    predictions: List[int],
    checkpoint_path: Path,
    device: torch.device,
    sample_count: int,
    vocab_size: int,
) -> str:
    """将测试指标整理成便于阅读和保存的文本报告。"""
    matrix = compute_confusion_matrix(labels, predictions, NUM_CLASSES)
    report = build_classification_report(labels, predictions, matrix)
    matrix_text = "\n".join(str(row) for row in matrix)

    lines = [
        "MultimodalModel 测试集评估",
        f"设备: {device}",
        f"Checkpoint: {checkpoint_path}",
        f"测试样本数: {sample_count}",
        f"词表大小: {vocab_size}",
        f"test_loss: {test_loss:.6f}",
        f"test_acc: {test_acc:.6f}",
        "",
        "各类别 precision / recall / f1-score:",
        report,
        "混淆矩阵:",
        "行表示真实标签，列表示预测标签",
        "类别顺序: " + ", ".join(CLASS_NAMES),
        matrix_text,
    ]
    return "\n".join(lines)


def main() -> None:
    """加载多模态模型的最佳 checkpoint，并在测试集上评估。"""
    ensure_output_dirs()

    torch_cache_dir = OUTPUT_DIR / "torch_cache"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache_dir))

    device = select_device()
    checkpoint_path = CHECKPOINT_DIR / "multimodal_best.pth"
    metrics_path = LOG_DIR / "multimodal_test_metrics.txt"

    checkpoint = load_checkpoint(checkpoint_path, device)
    vocab = checkpoint["vocab"]
    vocab_size = checkpoint.get("vocab_size", len(vocab))

    test_dataset = FoodDataset(
        metadata_path=TEST_METADATA,
        image_dir=TEST_IMAGE_DIR,
        vocab=vocab,
        train=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = MultimodalModel(
        vocab_size=vocab_size,
        num_classes=NUM_CLASSES,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

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
        vocab_size=vocab_size,
    )
    metrics_path.write_text(metrics_text + "\n", encoding="utf-8")

    print(metrics_text)
    print(f"\n测试指标已保存到: {metrics_path}")


if __name__ == "__main__":
    main()
