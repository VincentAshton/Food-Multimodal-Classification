"""测试多模态模型的前向传播。"""

import os

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, NUM_CLASSES, OUTPUT_DIR, TRAIN_IMAGE_DIR, TRAIN_METADATA
from dataset import FoodDataset
from models import MultimodalModel


def main() -> None:
    """从 DataLoader 读取一个 batch，验证 MultimodalModel 输出形状。"""
    # 将预训练权重缓存到项目目录，避免写入用户主目录导致权限问题。
    torch_cache_dir = OUTPUT_DIR / "torch_cache"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache_dir))

    train_dataset = FoodDataset(
        metadata_path=TRAIN_METADATA,
        image_dir=TRAIN_IMAGE_DIR,
        train=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MultimodalModel(
        vocab_size=len(train_dataset.vocab),
        num_classes=NUM_CLASSES,
        pretrained=True,
    )
    model.eval()

    batch = next(iter(train_loader))
    images = batch["image_tensor"]
    text_ids = batch["text_ids"]

    # 当前脚本只做前向传播和形状检查，不进行训练。
    with torch.no_grad():
        outputs = model(images, text_ids)

    print(f"图片输入形状: {tuple(images.shape)}")
    print(f"文本输入形状: {tuple(text_ids.shape)}")
    print(f"模型输出形状: {tuple(outputs.shape)}")

    expected_shape = (images.shape[0], NUM_CLASSES)
    assert tuple(outputs.shape) == expected_shape, (
        f"期望输出形状为 {expected_shape}，实际得到 {tuple(outputs.shape)}"
    )
    print("MultimodalModel 前向传播测试通过。")


if __name__ == "__main__":
    main()
