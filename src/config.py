"""Food_Multimodal 项目的统一配置文件。

本文件集中管理数据路径、类别名称、训练超参数、文本处理参数和输出目录。
其他脚本只需要导入这里的配置，避免在多个文件中重复写路径和参数。
"""

from pathlib import Path


# 项目根目录：src/config.py 的上一级目录。
ROOT_DIR = Path(__file__).resolve().parents[1]

# 数据集目录。
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# metadata.csv 路径，每个文件包含 image_path、label、text 三列。
TRAIN_METADATA = TRAIN_DIR / "metadata.csv"
VAL_METADATA = VAL_DIR / "metadata.csv"
TEST_METADATA = TEST_DIR / "metadata.csv"

# 图片目录路径。
TRAIN_IMAGE_DIR = TRAIN_DIR / "images"
VAL_IMAGE_DIR = VAL_DIR / "images"
TEST_IMAGE_DIR = TEST_DIR / "images"

# 五分类任务的类别名称，顺序会影响标签 id 的映射。
CLASS_NAMES = [
    "baozi",
    "cupcake",
    "french_fries",
    "rice",
    "youtiao",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
METADATA_COLUMNS = ["image_path", "label", "text"]

# 图像和训练相关超参数。
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
NUM_WORKERS = 2

# 文本处理相关参数。
MAX_TEXT_LEN = 40
MIN_WORD_FREQ = 1
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# 输出目录：保存模型权重、训练日志和曲线图。
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"


def ensure_output_dirs() -> None:
    """确保所有输出目录都已经创建。"""
    for directory in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
