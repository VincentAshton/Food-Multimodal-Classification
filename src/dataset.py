"""Food_Multimodal 项目的 PyTorch Dataset。

该模块负责从 metadata.csv 中读取 image_path、label、text，
并将图片和文本转换成后续模型可使用的张量形式。
"""

from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_TO_IDX, IMAGE_SIZE, MAX_TEXT_LEN, METADATA_COLUMNS, MIN_WORD_FREQ
from text_utils import build_vocab, encode_text


def build_image_transform(image_size: int = IMAGE_SIZE, train: bool = False) -> transforms.Compose:
    """构建默认图像预处理流程。"""
    transform_steps = [
        # 将所有图片缩放到统一尺寸，便于组成 batch。
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        # 训练阶段加入简单数据增强，提高模型泛化能力。
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.extend(
        [
            transforms.ToTensor(),
            # 使用 ImageNet 均值和标准差，适配预训练 ResNet18。
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(transform_steps)


class FoodDataset(Dataset):
    """食物图像-文本数据集，支持仅图像模型和多模态模型共用。"""

    def __init__(
        self,
        metadata_path: Path,
        image_dir: Path,
        transform: Optional[Callable] = None,
        vocab: Optional[Dict[str, int]] = None,
        max_text_len: int = MAX_TEXT_LEN,
        min_word_freq: int = MIN_WORD_FREQ,
        train: bool = False,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.image_dir = Path(image_dir)
        self.transform = transform if transform is not None else build_image_transform(train=train)
        self.max_text_len = max_text_len

        # 只读取项目需要的三列，避免 metadata 中额外列影响后续处理。
        self.data = pd.read_csv(self.metadata_path, usecols=METADATA_COLUMNS)
        self.data["text"] = self.data["text"].fillna("")

        # 训练集会创建词表；验证集和测试集应复用训练集词表。
        self.vocab = vocab if vocab is not None else build_vocab(self.data["text"], min_word_freq)

        unknown_labels = sorted(set(self.data["label"]) - set(CLASS_TO_IDX))
        if unknown_labels:
            raise ValueError(f"{self.metadata_path} 中存在未知类别: {unknown_labels}")

    def __len__(self) -> int:
        """返回当前数据集的样本数量。"""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """读取单个样本，并返回图片张量、文本 id 和标签 id。"""
        row = self.data.iloc[index]
        image_path = self.image_dir / row["image_path"]
        label_id = CLASS_TO_IDX[row["label"]]

        # 使用上下文管理器打开图片，避免文件句柄长期占用。
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        text_ids = encode_text(row["text"], self.vocab, self.max_text_len)

        return {
            "image_tensor": image_tensor,
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "label_id": torch.tensor(label_id, dtype=torch.long),
        }


# 保留这个别名，方便后续代码用更直观的多模态数据集名称导入。
FoodMultimodalDataset = FoodDataset
