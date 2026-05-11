"""仅测试图像分类模型的前向传播。"""

import os
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, NUM_CLASSES, OUTPUT_DIR, TRAIN_IMAGE_DIR, TRAIN_METADATA
from dataset import FoodDataset
from models import ImageOnlyModel


def main() -> None:
    """从训练集取一个 batch，验证 ImageOnlyModel 输出形状。"""
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

    model = ImageOnlyModel(num_classes=NUM_CLASSES, pretrained=True)
    model.eval()

    batch = next(iter(train_loader))

    # 当前测试只使用图像和标签，不使用 text_ids。
    images = batch["image_tensor"]
    labels = batch["label_id"]

    # 前向传播只检查模型结构和张量形状，不进行训练。
    with torch.no_grad():
        outputs = model(images)

    print(f"图片输入形状: {tuple(images.shape)}")
    print(f"模型输出形状: {tuple(outputs.shape)}")
    print(f"标签形状: {tuple(labels.shape)}")

    expected_shape = (images.shape[0], NUM_CLASSES)
    assert tuple(outputs.shape) == expected_shape, (
        f"期望输出形状为 {expected_shape}，实际得到 {tuple(outputs.shape)}"
    )
    print("ImageOnlyModel 前向传播测试通过。")


if __name__ == "__main__":
    main()
