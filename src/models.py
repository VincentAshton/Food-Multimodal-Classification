"""图像分类模型和图像-文本多模态分类模型定义。"""

import torch
from torch import nn
from torchvision import models

from config import NUM_CLASSES


class ImageOnlyModel(nn.Module):
    """基于 ResNet18 的仅图像输入分类模型。"""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()

        # 使用 torchvision 提供的 ResNet18，并按需加载 ImageNet 预训练权重。
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # 将原始 1000 分类输出层替换为当前任务的 5 分类输出层。
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """输入图片张量，输出每个类别的 logits。"""
        return self.backbone(image)


class MultimodalModel(nn.Module):
    """融合图像特征和文本特征的 5 分类模型。"""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = NUM_CLASSES,
        embedding_dim: int = 128,
        text_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # 图像分支：使用预训练 ResNet18，并去掉最后的分类 fc 层。
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.image_encoder = models.resnet18(weights=weights)
        image_feature_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()

        # 文本分支：Embedding 将 token id 转为词向量，LSTM 提取序列特征。
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=text_hidden_dim,
            batch_first=True,
        )

        # 融合分支：拼接图像特征和文本特征后进行最终分类。
        fusion_input_dim = image_feature_dim + text_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        """输入图片和文本 token id，输出每个类别的 logits。"""
        image_feature = self.image_encoder(images)

        embedded_text = self.embedding(text_ids)
        _, (hidden_state, _) = self.lstm(embedded_text)
        text_feature = hidden_state[-1]

        fused_feature = torch.cat([image_feature, text_feature], dim=1)
        return self.classifier(fused_feature)
