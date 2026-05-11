"""根据训练日志绘制 loss 和 accuracy 曲线。"""


import pandas as pd
import matplotlib.pyplot as plt

from config import FIGURE_DIR, LOG_DIR


def plot_curve(
    log_df: pd.DataFrame,
    y_columns: tuple[str, str],
    title: str,
    ylabel: str,
    output_name: str,
) -> None:
    """绘制两条指标曲线，并保存到 outputs/figures。"""
    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df[y_columns[0]], marker="o", label=y_columns[0])
    plt.plot(log_df["epoch"], log_df[y_columns[1]], marker="o", label=y_columns[1])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_name, dpi=300)
    plt.close()


def main() -> None:
    """读取训练日志，并分别保存损失曲线和准确率曲线。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 训练脚本会分别生成这两个 CSV 日志文件。
    image_log = pd.read_csv(LOG_DIR / "image_train_log.csv")
    multimodal_log = pd.read_csv(LOG_DIR / "multimodal_train_log.csv")

    plot_curve(
        image_log,
        ("train_loss", "val_loss"),
        "Image-only Loss Curve",
        "Loss",
        "image_loss_curve.png",
    )
    plot_curve(
        image_log,
        ("train_acc", "val_acc"),
        "Image-only Accuracy Curve",
        "Accuracy",
        "image_acc_curve.png",
    )
    plot_curve(
        multimodal_log,
        ("train_loss", "val_loss"),
        "Multimodal Loss Curve",
        "Loss",
        "multimodal_loss_curve.png",
    )
    plot_curve(
        multimodal_log,
        ("train_acc", "val_acc"),
        "Multimodal Accuracy Curve",
        "Accuracy",
        "multimodal_acc_curve.png",
    )

    print(f"训练曲线已保存到: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
