"""FoodDataset 的快速冒烟测试脚本。

该脚本只检查数据读取、文本编码、标签映射和 DataLoader 组 batch 是否正常，
不会训练或评估任何模型。
"""

from torch.utils.data import DataLoader

from config import IDX_TO_CLASS, TRAIN_IMAGE_DIR, TRAIN_METADATA
from dataset import FoodDataset


def print_sample(dataset: FoodDataset, index: int) -> None:
    """打印单个样本的关键字段。"""
    sample = dataset[index]
    row = dataset.data.iloc[index]
    label_id = sample["label_id"].item()

    print(f"\n样本 {index}")
    print(f"图片张量形状: {tuple(sample['image_tensor'].shape)}")
    print(f"文本 token id: {sample['text_ids'].tolist()}")
    print(f"标签 id: {label_id}")
    print(f"原始文本: {row['text']}")
    print(f"类别名称: {IDX_TO_CLASS[label_id]}")

    # FoodDataset 应返回适配 ResNet 的 RGB 图片张量，形状为 [通道数, 高度, 宽度]。
    assert tuple(sample["image_tensor"].shape) == (3, 224, 224), (
        "期望图片张量形状为 [3, 224, 224]，"
        f"实际得到 {tuple(sample['image_tensor'].shape)}"
    )


def main() -> None:
    """实例化训练集，并测试一个小 batch 的 DataLoader 输出。"""
    train_dataset = FoodDataset(
        metadata_path=TRAIN_METADATA,
        image_dir=TRAIN_IMAGE_DIR,
        train=False,
    )

    print(f"训练集长度: {len(train_dataset)}")

    # 查看前三个单独样本。
    for index in range(min(3, len(train_dataset))):
        print_sample(train_dataset, index)

    # 测试 PyTorch 默认 collate 能否正确处理 FoodDataset 返回的字典。
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
    )
    batch = next(iter(train_loader))

    print("\nBatch")
    print(f"图片 batch 形状: {tuple(batch['image_tensor'].shape)}")
    print(f"文本 batch 形状: {tuple(batch['text_ids'].shape)}")
    print(f"标签 batch 形状: {tuple(batch['label_id'].shape)}")

    assert tuple(batch["image_tensor"].shape[1:]) == (3, 224, 224), (
        "期望 batch 图片张量形状为 [batch_size, 3, 224, 224]，"
        f"实际得到 {tuple(batch['image_tensor'].shape)}"
    )


if __name__ == "__main__":
    main()
