import matplotlib.pyplot as plt

from typing import List

def plot_stats(
    train_loss: List[float],
    valid_loss: List[float],
    title: str
):
    plt.figure(figsize=(16, 8))

    plt.title(title + ' loss')

    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.grid()
    
    plt.show()


def plot_stats_dec(
    train_loss: List[float],
    valid_loss: List[float],

    train_acc: List[float],
    train_nmi: List[float],
    train_ari: List[float],
    val_acc: List[float],
    val_nmi: List[float],
    val_ari: List[float],

    title: str
):
    plt.figure(figsize=(16, 8))

    plt.title(title + " loss")

    plt.plot(train_loss, label="Train loss")
    plt.plot(valid_loss, label="Valid loss")
    plt.legend()
    plt.grid()

    plt.show()

    plt.figure(figsize=(16, 8))

    plt.title(title + " cluster accuracy, normalized mutual info score, adjusted random info score")

    plt.plot(train_acc, label="Train acc", linewidth=1.5)
    plt.plot(train_nmi, label="Train nmi", linestyle='dashdot')
    plt.plot(train_ari, label="Train ari", linestyle='dashed')

    plt.plot(val_acc, label="Valid acc", linewidth=1.5)
    plt.plot(val_nmi, label="Valid nmi", linestyle='dashdot')
    plt.plot(val_ari, label="Valid ari", linestyle='dashed')

    plt.legend()
    plt.grid()

    plt.show()

def plot_stats_idec(
    train_loss: List[float],
    valid_loss: List[float],
    
    train_rec_loss: List[float],
    
    train_cl_loss: List[float],

    train_acc: List[float],
    train_nmi: List[float],
    train_ari: List[float],
    val_acc: List[float],
    val_nmi: List[float],
    val_ari: List[float],

    title: str
):
    plt.figure(figsize=(16, 8))

    plt.title(title + " loss")

    plt.plot(train_loss, label="Train loss")
    plt.plot(valid_loss, label="Valid loss")
        
    plt.plot(train_rec_loss, label="Train Reconstruct. loss",linestyle='dashdot')
    plt.plot(train_cl_loss, label="Train Clustering loss",linestyle='dashed')
    plt.legend()
    plt.grid()

    plt.show()

    plt.figure(figsize=(16, 8))

    plt.title(title + " cluster accuracy, normalized mutual info score, adjusted random info score")

    plt.plot(train_acc, label="Train acc", linewidth=2)
    plt.plot(train_nmi, label="Train nmi", linestyle='dashdot')
    plt.plot(train_ari, label="Train ari", linestyle='dashed')

    plt.plot(val_acc, label="Valid acc", linewidth=2)
    plt.plot(val_nmi, label="Valid nmi", linestyle='dashdot')
    plt.plot(val_ari, label="Valid ari", linestyle='dashed')

    plt.legend()
    plt.grid()

    plt.show()
    