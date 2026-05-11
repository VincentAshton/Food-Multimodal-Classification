"""数据集检查脚本。

用于检查 train/val/test 三个划分下的 images 目录和 metadata.csv 是否存在，
并统计每个划分的样本数量和类别分布。
"""

from pathlib import Path

import pandas as pd

from config import CLASS_NAMES, DATA_DIR


# metadata.csv 必须包含这三列。
REQUIRED_COLUMNS = {"image_path", "label", "text"}


def check_split(split_name: str) -> None:
    """检查单个数据划分，并打印基本统计信息。"""
    split_dir = DATA_DIR / split_name
    image_dir = split_dir / "images"
    metadata_path = split_dir / "metadata.csv"

    print(f"\n[{split_name}]")
    print(f"images dir: {image_dir} - {'OK' if image_dir.exists() else 'MISSING'}")
    print(f"metadata:   {metadata_path} - {'OK' if metadata_path.exists() else 'MISSING'}")

    if not metadata_path.exists():
        return

    df = pd.read_csv(metadata_path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        print(f"缺少必要列: {sorted(missing_columns)}")
        return

    print(f"样本数量: {len(df)}")
    print("类别分布:")
    counts = df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    for label, count in counts.items():
        print(f"  {label}: {count}")

    # 如果 metadata 中出现了配置文件未登记的类别，需要及时提醒。
    unknown_labels = sorted(set(df["label"]) - set(CLASS_NAMES))
    if unknown_labels:
        print(f"未知类别: {unknown_labels}")


def main() -> None:
    """依次检查 train、val、test 三个数据划分。"""
    print(f"数据集根目录: {Path(DATA_DIR)}")
    for split_name in ["train", "val", "test"]:
        check_split(split_name)


if __name__ == "__main__":
    main()
