import torch
from torch import nn
import numpy as np
import json
import time
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import seaborn as sns

# ============================
# IMPORTA TUS ATAQUES AQUÍ
# ============================
from ImageNet100ValDataset import * 
from ataqueFGSM2 import * 
from ataquePDG2 import *
from ataqueRFGSM2 import *
from AtaquesGaussianos2 import *


# ============================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================
def show_curves(all_curves):

    final_curve_means = {k: np.mean([c[k] for c in all_curves], axis=0) for k in all_curves[0].keys()}
    final_curve_stds = {k: np.std([c[k] for c in all_curves], axis=0) for k in all_curves[0].keys()}

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.set_facecolor('white')

    epochs = np.arange(len(final_curve_means["val_loss"])) + 1

    ax[0].plot(epochs, final_curve_means['val_loss'], label='validation')
    ax[0].plot(epochs, final_curve_means['train_loss'], label='training')
    ax[0].fill_between(epochs, y1=final_curve_means["val_loss"] - final_curve_stds["val_loss"], 
                             y2=final_curve_means["val_loss"] + final_curve_stds["val_loss"], 
                             alpha=.5)
    ax[0].fill_between(epochs, y1=final_curve_means["train_loss"] - final_curve_stds["train_loss"], 
                             y2=final_curve_means["train_loss"] + final_curve_stds["train_loss"], 
                             alpha=.5)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss evolution during training')
    ax[0].legend()

    ax[1].plot(epochs, final_curve_means['val_acc'], label='validation')
    ax[1].plot(epochs, final_curve_means['train_acc'], label='training')
    ax[1].fill_between(epochs, y1=final_curve_means["val_acc"] - final_curve_stds["val_acc"], 
                             y2=final_curve_means["val_acc"] + final_curve_stds["val_acc"], 
                             alpha=.5)
    ax[1].fill_between(epochs, y1=final_curve_means["train_acc"] - final_curve_stds["train_acc"], 
                             y2=final_curve_means["train_acc"] + final_curve_stds["train_acc"], 
                             alpha=.5)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy evolution during training')
    ax[1].legend()

    plt.show()


# ============================================================
# APLICA LOS ATAQUES SEGÚN PROPORCIONES
# ============================================================
def apply_attacks_on_batch(x, y, model, attack_params):
    """
    Mezcla ataques en el batch según proporciones.
    x: imágenes del batch
    y: labels
    attack_params: diccionario {'fgsm':0.3, 'pgd':0.2, ...}
    """
    batch_size = x.shape[0]
    device = x.device

    idx = torch.randperm(batch_size, device=device)
    x_adv = x.clone()

    start = 0
    for attack_name, proportion in attack_params.items():
        n = int(batch_size * proportion)
        if n == 0:
            continue

        indices = idx[start:start+n]


        if attack_name == "fgsm":
            x_adv[indices] = generar_imagen_fgsm(model, x_adv[indices], y[indices], param_values[attack_name])

        elif attack_name == "pgd":
            x_adv[indices] = generar_imagen_PGD(model, x_adv[indices], y[indices], param_values[attack_name])

        elif attack_name == "rfgsm":
            x_adv[indices] = generar_imagen_rfgsm(model, x_adv[indices], y[indices], param_values[attack_name])

        elif attack_name == "gauss":
            x_adv[indices] = generar_imagen_gaussian(model, x_adv[indices], y[indices],param_values[attack_name])

        elif attack_name == "clean":
            pass  # No ataque

        else:
            raise ValueError(f"Ataque desconocido: {attack_name}")

        start += n

    return x_adv, y


# ============================================================
# TRAIN STEP CON ATAQUES
# ============================================================
def train_step(x_batch, y_batch, model, optimizer, criterion, attack_params, use_gpu):
    
    # Aplicar ataques
    x_batch, y_batch = apply_attacks_on_batch(x_batch, y_batch, model, attack_params)

    # Predicción
    y_predicted = model(x_batch)

    # Cálculo de loss
    loss = criterion(y_predicted, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return y_predicted, loss


# ============================================================
# EVALUACIÓN
# ============================================================
def evaluate(val_loader, model, criterion, use_gpu):
    cumulative_loss = 0
    cumulative_predictions = 0
    data_count = 0

    for x_val, y_val in val_loader:
        if use_gpu:
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        y_predicted = model(x_val)

        loss = criterion(y_predicted, y_val)

        class_prediction = torch.argmax(y_predicted, axis=1).long()

        cumulative_predictions += (y_val == class_prediction).sum().item()
        cumulative_loss += loss.item() * y_val.shape[0]
        data_count += y_val.shape[0]

    val_acc = cumulative_predictions / data_count
    val_loss = cumulative_loss / data_count

    return val_acc, val_loss


# ============================================================
# TRAIN MODEL COMPLETO
# ============================================================
def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs,
    criterion,
    batch_size,
    lr,
    attack_params,
    n_evaluations_per_epoch=6,
    use_gpu=False,
):

    if use_gpu:
        model.cuda()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_gpu
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=use_gpu
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    curves = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    t0 = time.perf_counter()

    iteration = 0
    n_batches = len(train_loader)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        cumulative_train_loss = 0
        cumulative_train_corrects = 0
        examples_count = 0

        model.train()
        for i, (x_batch, y_batch) in enumerate(train_loader):
            if use_gpu:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            y_predicted, loss = train_step(
                x_batch, y_batch,
                model,
                optimizer,
                criterion,
                attack_params,
                use_gpu
            )

            cumulative_train_loss += loss.item() * x_batch.shape[0]
            examples_count += y_batch.shape[0]

            class_prediction = torch.argmax(y_predicted, axis=1).long()
            cumulative_train_corrects += (y_batch == class_prediction).sum().item()

            if (i % max(1, (n_batches // n_evaluations_per_epoch)) == 0) and (i > 0):
                train_loss = cumulative_train_loss / examples_count
                train_acc = cumulative_train_corrects / examples_count
                print(f" Iter {iteration} - Batch {i}/{len(train_loader)} "
                      f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")

            iteration += 1

        model.eval()
        with torch.no_grad():
            val_acc, val_loss = evaluate(val_loader, model, criterion, use_gpu)

        print(f" Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        train_loss = cumulative_train_loss / examples_count
        train_acc = cumulative_train_corrects / examples_count

        curves["train_acc"].append(train_acc)
        curves["val_acc"].append(val_acc)
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

    total_time = time.perf_counter() - t0
    print(f"\nTiempo total de entrenamiento: {total_time:.4f} [s]")

    model.cpu()

    return curves, total_time


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    from torchvision.models import resnet34, ResNet34_Weights

    use_gpu = False

    train_dataset = ImageNet100ValDataset("PGD_untargeted_out", transform=transform)
    val_dataset = ImageNet100ValDataset("3 clases", transform=transform)

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)

    lr = 5e-4
    batch_size = 32
    criterion = nn.CrossEntropyLoss()
    epochs = 5

    # ======================================================
    # PROPORCIONES DE ATAQUES
    # LA SUMA DEBE SER 1.0
    # ======================================================
    attack_params = {
        "fgsm": 0.3,
        "pgd": 0.2,
        "rfgsm": 0.2,
        "gauss": 0.2,
        "clean": 0.1
    }
    param_values = {
            "fgsm": 0.05,
            "pgd": [0.05, 2/255, 5],
            "rfgsm": [0.05, 2/255],
            "gauss": [0.05],
            "clean": 0
        }
    all_curves = []
    times = []

    curves, total_time = train_model(
        model,
        train_dataset,
        val_dataset,
        epochs,
        criterion,
        batch_size,
        lr,
        attack_params,
        use_gpu=use_gpu,
    )

    all_curves.append(curves)
    times.append(total_time)

    print(f"\nTiempo promedio: {np.mean(times):.2f} ± {np.std(times):.2f} [s]")

    show_curves(all_curves)

    torch.save(model.state_dict(), "resnet34_retrained.pth")
