"""多模态模型使用的文本预处理工具。

包括英文分词、词表构建、token id 转换和定长 padding。
"""

from collections import Counter
import re
from typing import Dict, Iterable, List, Sequence

from config import PAD_TOKEN, UNK_TOKEN


# 简单英文 token 规则：保留英文单词、带撇号的缩写和数字。
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")


def tokenize(text: str) -> List[str]:
    """将英文描述转成小写 token 列表。"""
    if not isinstance(text, str):
        text = ""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(texts: Iterable[str], min_freq: int = 1) -> Dict[str, int]:
    """根据训练文本构建 token 到 id 的词表。"""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    # 0 留给 padding，1 留给未登录词，方便 Embedding 使用 padding_idx=0。
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in sorted(counter.items()):
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def numericalize(tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    """将 token 序列转换为整数 id 序列。"""
    unk_idx = vocab.get(UNK_TOKEN, 1)
    return [vocab.get(token, unk_idx) for token in tokens]


def pad_sequence(sequence: List[int], max_len: int, pad_value: int = 0) -> List[int]:
    """将 token id 序列截断或补齐到固定长度。"""
    sequence = sequence[:max_len]
    return sequence + [pad_value] * (max_len - len(sequence))


def encode_text(
    text: str,
    vocab: Dict[str, int],
    max_len: int,
    pad_token: str = PAD_TOKEN,
) -> List[int]:
    """完成单条文本的分词、id 转换和 padding。"""
    pad_value = vocab.get(pad_token, 0)
    token_ids = numericalize(tokenize(text), vocab)
    return pad_sequence(token_ids, max_len=max_len, pad_value=pad_value)
