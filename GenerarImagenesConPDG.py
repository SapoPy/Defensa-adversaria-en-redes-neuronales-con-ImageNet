import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path
import json

from torchvision.models import resnet34, ResNet34_Weights
from ImageNet100ValDataset import *  # tu clase

# --- constantes ImageNet ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --------------------------
# PGD (funciona con tensores normalizados)
# --------------------------
def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10,
               random_start=True, targeted=False, device=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()
    images = images.to(device)
    labels = labels.to(device)

    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, -1, 1, 1)
    std_tensor = std_t

    # eps/alpha en *espacio normalizado* (dividir por std por canal)
    eps_norm = (eps / std_tensor).to(device)
    alpha_norm = (alpha / std_tensor).to(device)

    min_norm = ((0.0 - mean_t) / std_t)
    max_norm = ((1.0 - mean_t) / std_t)

    x_orig = images.clone().detach()
    x_adv = x_orig.clone().detach()
    if random_start:
        rand_pert = torch.empty_like(x_adv).uniform_(-eps, eps).to(device)
        rand_pert_norm = rand_pert / std_tensor
        x_adv = torch.clamp(x_adv + rand_pert_norm, min_norm, max_norm).detach()

    x_adv.requires_grad_(True)

    for _ in range(iters):
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, labels)
        if targeted:
            loss = -loss

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()
        grad = x_adv.grad.data

        if targeted:
            x_adv = x_adv - alpha_norm * grad.sign()
        else:
            x_adv = x_adv + alpha_norm * grad.sign()

        # proyectar L_inf en espacio normalizado alrededor de x_orig
        x_adv = torch.max(torch.min(x_adv, x_orig + eps_norm), x_orig - eps_norm)
        x_adv = torch.max(torch.min(x_adv, max_norm), min_norm).detach()
        x_adv.requires_grad_(True)

    return x_adv.detach()

# --------------------------
# util: convertir WNID -> índice en vector de 1000 clases del modelo
# --------------------------
def wnid_to_imagenet_index(target_wnid, weights, labels_json_path="Labels.json"):
    with open(labels_json_path) as f:
        wnid2name = json.load(f)
    target_name = wnid2name[target_wnid].split(',')[0].strip()
    imagenet_classes = weights.meta["categories"]
    # buscar por coincidencia más robusta (lower)
    for i, cname in enumerate(imagenet_classes):
        if cname.lower().strip() == target_name.lower().strip():
            return i
    # si no encuentra, intenta coincidencia parcial
    for i, cname in enumerate(imagenet_classes):
        if target_name.lower().split()[0] in cname.lower():
            return i
    raise ValueError(f"No pude mapear {target_wnid} -> nombre ImageNet ({target_name}). Revisa Labels.json y weights.meta['categories'].")

# --------------------------
# Generar adversarios y guardarlos en disco manteniendo estructura WNID/
# Devuelve (out_dir_path, ImageNet100ValDataset apuntando a out_dir)
# --------------------------
def generar_dataset_pgd_guardado_como_dataset(model, dataset, out_dir,
                                               eps=8/255, alpha=2/255, iters=10,
                                               batch_size=32, device=None, overwrite=False,
                                               mean=IMAGENET_MEAN, std=IMAGENET_STD,
                                               targeted=False, target_idx=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, -1, 1, 1)

    idx_global = 0
    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)

        if targeted:
            if target_idx is None:
                raise ValueError("target_idx required for targeted=True")
            target_labels = torch.full((imgs.size(0),), target_idx, device=device, dtype=torch.long)
            imgs_adv = pgd_attack(model, imgs, target_labels, eps=eps, alpha=alpha, iters=iters,
                                  random_start=True, targeted=True, device=device, mean=mean, std=std)
        else:
            imgs_adv = pgd_attack(model, imgs, labels, eps=eps, alpha=alpha, iters=iters,
                                  random_start=True, targeted=False, device=device, mean=mean, std=std)

        imgs_adv_pixel = (imgs_adv * std_t + mean_t).clamp(0.0, 1.0).cpu()

        # guardar manteniendo wnid de la muestra original (usa dataset.samples)
        for b in range(imgs_adv_pixel.size(0)):
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

    # crear ImageNet100ValDataset apuntando a out_dir, reusando transform y labels_json si existen
    try:
        adv_ds = ImageNet100ValDataset(str(out_dir), transform=dataset.transform, labels_json=getattr(dataset, "labels_json", "Labels.json"))
    except TypeError:
        # si tu clase no espera labels_json en constructor
        adv_ds = ImageNet100ValDataset(str(out_dir), transform=dataset.transform)
    return str(out_dir), adv_ds

# --------------------------
# medir targeted success (fracción de muestras predichas = target_idx)
# admite ImageNet100ValDataset o cualquier dataset que devuelva (img_tensor_normalized, label_idx)
# --------------------------
def measure_targeted_success(model, adv_dataset, target_idx, batch_size=32, device=None):
    """
    Mide la tasa de éxito de un ataque dirigido:
    - adv_dataset: debe devolver (img_tensor_normalized, label_idx) como ImageNet100ValDataset
    - target_idx: índice de clase del modelo 1000 al que se quiere dirigir el ataque
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()
    loader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=False)

    total = 0
    success = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)  # shape (B, 1000)
            preds = outputs.argmax(dim=1)  # top-1 clase predicha
            success += (preds == target_idx).sum().item()
            total += imgs.size(0)

    return success / total if total > 0 else 0.0

# --------------------------
# evaluar top1/top5 — asume que dataset devuelve (img_norm, local_label)
# Si quieres evaluar sobre las 100 clases originales, pasa selected_indices_in_model y selected_classes (wnids)
# --------------------------
def evaluate_model(model, dataset, batch_size=32, device=None, weights=None, labels_json="Labels.json"):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # construir remapeo local->model1000 si dataset has class_to_idx and weights provided
    map_local_to_model = None
    selected_indices_in_model = None
    if hasattr(dataset, "class_to_idx") and weights is not None:
        # selected_classes wnids en orden local
        selected_classes = list(dataset.class_to_idx.keys())
        # crear wnid->name
        with open(labels_json) as f:
            wnid2name = json.load(f)
        imagenet_classes = weights.meta["categories"]
        # map each wnid -> index in imagenet_classes using first token of name
        selected_indices_in_model = []
        for wnid in selected_classes:
            name = wnid2name[wnid].split(',')[0].strip()
            idx = None
            for i, cname in enumerate(imagenet_classes):
                if cname.lower().strip() == name.lower().strip():
                    idx = i; break
            if idx is None:
                # fallback partial match
                for i, cname in enumerate(imagenet_classes):
                    if name.lower().split()[0] in cname.lower():
                        idx = i; break
            if idx is None:
                raise ValueError(f"No pude mapear wnid {wnid} -> imagenet name {name}")
            selected_indices_in_model.append(idx)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for imgs, local_labels in loader:
            imgs = imgs.to(device)
            local_labels = local_labels.to(device)

            outs = model(imgs)  # shape (B,1000)

            if selected_indices_in_model is not None:
                outs = outs[:, selected_indices_in_model]  # (B, #local_classes)

            # top1/topk on outs
            top1 = outs.argmax(dim=1)
            k = min(5, outs.size(1))
            topk = torch.topk(outs, k, dim=1).indices

            # NOTE: if we filtered to local classes, local_labels are already indices in that local space
            # if not filtered, local_labels must be model-level indices (0..999)
            if selected_indices_in_model is not None:
                correct_top1 += (top1 == local_labels).sum().item()
                for i in range(local_labels.size(0)):
                    if local_labels[i].item() in topk[i]:
                        correct_top5 += 1
            else:
                correct_top1 += (top1 == local_labels).sum().item()
                for i in range(local_labels.size(0)):
                    if local_labels[i].item() in topk[i]:
                        correct_top5 += 1

            total += local_labels.size(0)

    acc_top1 = correct_top1 / total if total>0 else 0.0
    acc_top5 = correct_top5 / total if total>0 else 0.0
    return acc_top1, acc_top5

# --------------------------
# example main
# --------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device).eval()

    # dataset original (carpeta con subfolders wnid/)
    src_dir = "3 clases"   # ajusta
    ds = ImageNet100ValDataset(src_dir, transform=transform)

    # target WNID que queremos forzar (ejemplo)
    target_wnid = "n01514859"
    target_idx = wnid_to_imagenet_index(target_wnid, weights, labels_json_path="Labels.json")
    print("Target index in model outputs:", target_idx)

    # generar dataset adversarial dirigido y devolver un ImageNet100ValDataset apuntando a la carpeta nueva
    out_dir, adv_ds = generar_dataset_pgd_guardado_como_dataset(model, ds, out_dir="val_pgd_targeted_015",
                                                                eps=0.15, alpha=2/255, iters=10,
                                                                batch_size=32, device=device,
                                                                targeted=True, target_idx=target_idx)

    # medir targeted success (fracción de predicciones exactas = target_idx en el espacio 1000)
    succ = measure_targeted_success(model, adv_ds, target_idx, batch_size=32, device=device)
    print(f"Targeted success rate for {target_wnid}: {100*succ:.2f}%")

    # evaluar top1/top5 sobre el dataset adversarial (usa mapeo local->model si adv_ds tiene class_to_idx)
    acc1, acc5 = evaluate_model(model, adv_ds, batch_size=32, device=device, weights=weights, labels_json="Labels.json")
    print(f"Acc on adv dataset (top1/top5): {acc1:.4f}, {acc5:.4f}")
