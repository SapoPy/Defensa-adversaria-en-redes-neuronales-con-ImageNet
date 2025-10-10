import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from pathlib import Path
from ImageNet100ValDataset import *

# ImageNet stats por defecto
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------- PGD attack (batch) ----------
def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10,
               random_start=True, targeted=False, device=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    images: tensor (B,C,H,W) ya NORMALIZADO si el modelo espera normalizado.
            (Si tienes images en pixel-space, normalízalas antes de llamar aquí.)
    labels: tensor (B,) índice de clase
    eps: máximo perturbación en pixel-space (ej. 8/255)
    alpha: step size en pixel-space (ej. 2/255)
    iters: nº de iteraciones PGD
    random_start: si True hace start aleatorio en L_inf ball
    targeted: si True hace ataque dirigido (necesitas suministrar target labels en 'labels')
    Devuelve: perturbed images (B,C,H,W) en el MISMO espacio que images (normalizado)
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()
    images = images.to(device)
    labels = labels.to(device)

    # preparar constantes en espacio normalizado
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, -1, 1, 1)

    # convertir eps, alpha a escala normalizada: eps_norm = eps / std_channel
    std_tensor = torch.tensor(std, device=device).view(1, -1, 1, 1)
    eps_norm = (eps / std_tensor).to(device)
    alpha_norm = (alpha / std_tensor).to(device)

    # límites en espacio normalizado (para proyectar)
    min_norm = ((0.0 - mean_t) / std_t)
    max_norm = ((1.0 - mean_t) / std_t)

    # init
    x_adv = images.clone().detach()
    if random_start:
        # random uniform in [-eps, eps] in pixel space -> convert to normalized
        rand_pert = (torch.empty_like(x_adv).uniform_(-eps, eps)).to(device)
        rand_pert_norm = rand_pert / std_tensor
        x_adv = torch.clamp(x_adv + rand_pert_norm, min_norm, max_norm).detach()

    x_adv.requires_grad_(True)

    for i in range(iters):
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, labels)
        if targeted:
            # targeted: minimize loss towards target class -> use negative grad of loss
            loss = -loss

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()
        grad = x_adv.grad.data

        # step
        if targeted:
            x_adv = x_adv - alpha_norm * grad.sign()
        else:
            x_adv = x_adv + alpha_norm * grad.sign()

        # project back to L_inf ball around original images (in normalized space)
        # ensure: x_adv in [images - eps_norm, images + eps_norm]
        x_adv = torch.max(torch.min(x_adv, images + eps_norm), images - eps_norm)
        # clamp to valid normalized range
        x_adv = torch.max(torch.min(x_adv, max_norm), min_norm).detach()
        x_adv.requires_grad_(True)

    return x_adv.detach()

# ---------- crear y guardar dataset PGD ----------
# ---------- generar_dataset_pgd_y_guardar (corregido) ----------
def generar_dataset_pgd_y_guardar(model, dataset, out_dir, eps=8/255, alpha=2/255, iters=10,
                                  batch_size=32, device=None, overwrite=False,
                                  mean=IMAGENET_MEAN, std=IMAGENET_STD,
                                  targeted=False, target_idx=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, -1, 1, 1)

    adv_tensors = []
    adv_labels = []
    idx_global = 0

    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)

        # si es targeted creamos etiquetas objetivo (índice en 0..999)
        if targeted:
            if target_idx is None:
                raise ValueError("target_idx required for targeted=True")
            target_labels = torch.full((imgs.size(0),), target_idx, device=device, dtype=torch.long)
            imgs_adv = pgd_attack(model, imgs, target_labels,
                                  eps=eps, alpha=alpha, iters=iters,
                                  random_start=True, targeted=True,
                                  device=device, mean=mean, std=std)
        else:
            imgs_adv = pgd_attack(model, imgs, labels,
                                  eps=eps, alpha=alpha, iters=iters,
                                  random_start=True, targeted=False,
                                  device=device, mean=mean, std=std)

        # guardado y append (mantener adv_labels = original labels para análisis)
        adv_tensors.append(imgs_adv.cpu())
        adv_labels.append(labels.cpu())

        # guardar denormalizadas (si dataset original tiene .samples, sacamos wnid por índice)
        imgs_adv_pixel = (imgs_adv * std_t + mean_t).clamp(0.0, 1.0).cpu()
        for b in range(imgs_adv_pixel.size(0)):
            # Intentar obtener ruta original (si dataset tiene attribute samples)
            try:
                orig_path = dataset.samples[idx_global + b][0]
                wnid = Path(orig_path).parent.name
                fname = Path(orig_path).name
            except Exception:
                wnid = str(labels[b].item())
                fname = f"adv_{idx_global + b:06d}.png"

            class_dir = out_dir / wnid; class_dir.mkdir(parents=True, exist_ok=True)
            save_path = class_dir / f"{Path(fname).stem}_adv.png"
            if overwrite or not save_path.exists():
                save_image(imgs_adv_pixel[b], save_path)

        idx_global += imgs.size(0)

    adv_tensors = torch.cat(adv_tensors, dim=0)
    adv_labels = torch.cat(adv_labels, dim=0)

    # Devolver TensorDataset (normalizado) — es lo más sencillo y seguro
    tensor_ds = TensorDataset(adv_tensors, adv_labels)
    return str(out_dir), tensor_ds

def measure_targeted_success(model, adv_tensor_ds, target_idx, batch_size=32, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(adv_tensor_ds, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    total = 0; success = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outs = model(imgs)
            preds = outs.argmax(dim=1)
            success += (preds == target_idx).sum().item()
            total += imgs.size(0)
    return success / total

import json

def wnid_to_imagenet_index(target_wnid, weights, labels_json_path="Labels.json"):
    # labels: WNID -> "nombre, alias, ..."
    with open(labels_json_path) as f:
        wnid2name = json.load(f)
    # tomamos la primera palabra/frase antes de la coma
    target_name = wnid2name[target_wnid].split(',')[0].strip()
    imagenet_classes = weights.meta["categories"]   # lista de 1000 nombres (strings)
    target_idx = imagenet_classes.index(target_name)  # ValueError si no encuentra
    return target_idx

# ---------- evaluación simple (top1/top5) ----------
def evaluate_model(model, dataset, batch_size=32, device=None):
    """
    Evalúa top1/top5 sobre un Dataset o TensorDataset que devuelve (img_tensor_normalized, label_idx).

    Si se pasan selected_indices_in_model y selected_classes, se asume que las etiquetas son índices locales.
    Si no, se evalúa directamente sobre las 1000 clases originales.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()

    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for imgs, labels_idx in val_loader:
            # Inferencia
            outputs = model(imgs)

            # Filtrar solo tus 100 clases
            filtered_logits = outputs[:, selected_indices_in_model]
            filtered_probs = torch.nn.functional.softmax(filtered_logits, dim=1)

            # Predicciones
            preds_in_filtered = filtered_probs.argmax(dim=1)
            
            top5_preds = torch.topk(filtered_probs, 5, dim=1).indices

            # Mapeo de índice filtrado → wnid real
            pred_wnids = [selected_classes[i] for i in preds_in_filtered]
            true_wnids = [list(dataset.class_to_idx.keys())[i] for i in labels_idx]
            
            # Sumar aciertos
            for i in range(labels_idx.size(0)):
                if labels_idx[i].item() in top5_preds[i]:
                    correct_top5 += 1
                    
            correct_top1 += sum(p == t for p, t in zip(pred_wnids, true_wnids))
            total += len(imgs)

    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total

    return acc_top1, acc_top5

if __name__ == "__main__":
    from torchvision.models import resnet34, ResNet34_Weights
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).eval()

    ds = ImageNet100ValDataset("3 clases", transform=weights.transforms(), labels_json="Labels.json")

    # 1) convertir WNID a índice model (1000 clases)
    target_wnid = "n01514859" 
    target_idx = wnid_to_imagenet_index(target_wnid, weights, labels_json_path="Labels.json")
    print("Target index in model outputs:", target_idx)

    # 2) generar dataset PGD dirigido al target_idx
    out_dir, tensor_ds = generar_dataset_pgd_y_guardar(model, ds, out_dir="val_pgd_targeted",
                                                      eps= 0.05, alpha=2/255, iters=10,
                                                      batch_size=32, device=None,
                                                      targeted=True, target_idx=target_idx)

    # 3) medir targeted success
    succ = measure_targeted_success(model, tensor_ds, target_idx, batch_size=32)
    print(f"Targeted success rate for {target_wnid}: {100*succ:.2f}%")

    # 4) además mide accuracy normal/top5 si quieres:
    acc1, acc5 = evaluate_model(model, tensor_ds, batch_size=32)  # normal eval (no filtering)
    print("Acc on adv dataset (top1/top5):", acc1, acc5)
