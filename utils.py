import torch
import torch.nn.functional as F
from pathlib import Path
import json
from PIL import Image
from torchvision.models import resnet34, ResNet34_Weights
from ImageNet100ValDataset import *

# ------------------ Estadísticas de ImageNet ------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def image_gradient(model, img_norm, label, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()

    x = img_norm.unsqueeze(0).to(device)
    x.requires_grad_(True)

    y = torch.tensor([label], device=device)
    out = model(x)
    loss = F.cross_entropy(out, y)

    loss.backward()
    return x.grad.detach().squeeze(0)   # grad NORMALIZADO

# ------------------ utilidades ------------------
def denormalize_tensor(tensor_norm, mean=MEAN, std=STD):
    """tensor_norm: normalizado -> pixel-space [0,1]"""
    m = torch.tensor(mean, device=tensor_norm.device).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=tensor_norm.device).view(1, -1, 1, 1)
    return (tensor_norm * s + m).clamp(0.0, 1.0)

def normalize_tensor(px, mean=MEAN, std=STD):
    m = torch.tensor(mean, device=px.device).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=px.device).view(1, -1, 1, 1)
    return (px - m) / s

def evaluar_imagen(model, img, selected_indices_in_model):
    """
    Evalúa una sola imagen en el modelo.

    Parámetros:
        model: modelo preentrenado (por ejemplo, resnet50)
        img: tensor de imagen (C, H, W) ya transformado
        selected_indices_in_model: lista de índices de las clases que se quieren evaluar en el modelo

    Devuelve:
        pred_wnid: WNID predicho
        nombre_legible: nombre de la clase predicha
        prob: probabilidad asociada
    """

    input_tensor = img.unsqueeze(0)

    # Inferencia sin gradientes
    with torch.no_grad():
        output = model(input_tensor)

        # Filtrar los logits solo para las clases seleccionadas
        filtered_logits = output[0][selected_indices_in_model]
        filtered_probs = torch.nn.functional.softmax(filtered_logits, dim=0)

        # Elegir la clase más probable
        pred_idx_in_filtered = filtered_probs.argmax().item()
        pred_wnid = selected_classes[pred_idx_in_filtered]
        prob = filtered_probs[pred_idx_in_filtered].item()

        nombre_legible = labels[pred_wnid]

    return pred_wnid, nombre_legible, prob

def load_resnet34(device=None):
    """
    Carga un modelo ResNet34 preentrenado con pesos de ImageNet,
    listo para evaluación, junto con su lista de clases.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device).eval()
    imagenet_classes = weights.meta["categories"]
    preprocess = weights.transforms()
    return model, imagenet_classes, preprocess


# =====================================================
# MAPEO ENTRE WNID Y CLASES DEL MODELO
# =====================================================

def load_wnid_mapping(labels_json_path="Labels.json"):
    """Carga el mapeo WNID → nombre legible desde un JSON."""
    with open(labels_json_path, "r") as f:
        wnid2name = json.load(f)
    return wnid2name


def build_class_mapping(dataset_root, imagenet_classes, wnid2name):
    """
    Construye un mapeo robusto entre las carpetas locales de ImageNet100
    y los índices de clase del modelo ResNet34 (1000 clases).

    Retorna:
        wnid_to_model_idx, model_idx_to_wnid, selected_indices_in_model
    """
    dataset_root = Path(dataset_root)
    selected_classes = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])

    wnid_to_model_idx = {}
    for wnid in selected_classes:
        if wnid not in wnid2name:
            continue
        wnid_name = wnid2name[wnid].split(",")[0].lower().strip()
        match = [i for i, c in enumerate(imagenet_classes) if wnid_name in c.lower()]
        if match:
            wnid_to_model_idx[wnid] = match[0]

    selected_indices_in_model = list(wnid_to_model_idx.values())
    model_idx_to_wnid = {v: k for k, v in wnid_to_model_idx.items()}

    print(f"Total carpetas detectadas en {dataset_root}: {len(selected_classes)}")
    print(f"Total clases mapeadas correctamente: {len(selected_indices_in_model)}")

    if not selected_indices_in_model:
        raise ValueError("No se encontró ninguna clase del dataset en las 1000 del modelo. Verifica Labels.json.")

    return wnid_to_model_idx, model_idx_to_wnid, selected_indices_in_model


# =====================================================
# EVALUACIÓN TOP-5 LIMITADA A 100 CLASES
# =====================================================

def top5_for_image_path(model, img_path, preprocess, imagenet_classes,
                        selected_indices_in_model, model_idx_to_wnid, wnid2name,
                        device=None):
    """
    Evalúa una imagen (ruta) y retorna las top-5 predicciones restringidas
    a las clases seleccionadas del dataset (p.ej. las 100 de ImageNet100).

    Devuelve una lista de dicts con:
    - model_idx
    - class_name
    - prob
    - wnid
    - wnid_name
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)[0]
        logits_filtered = logits[selected_indices_in_model]
        probs = F.softmax(logits_filtered, dim=0)

    k = min(5, len(selected_indices_in_model))
    topk = torch.topk(probs, k=k)

    results = []
    for idx_f, p in zip(topk.indices.tolist(), topk.values.tolist()):
        model_idx = selected_indices_in_model[idx_f]
        wnid = model_idx_to_wnid.get(model_idx, "unknown")
        wnid_name = wnid2name.get(wnid, "desconocido")
        class_name = imagenet_classes[model_idx]
        results.append({
            "model_idx": model_idx,
            "class_name": class_name,
            "prob": p,
            "wnid": wnid,
            "wnid_name": wnid_name
        })
    return results

import json
from torchvision.models import resnet34, ResNet34_Weights
import torch

def wnid_to_model_index(wnid, weights, labels_json="Labels.json"):
    """
    Devuelve el índice (0..999) en weights.meta['categories'] correspondiente al wnid.
    Intenta coincidencia exacta con el nombre "primera parte" del Labels.json,
    y si no encuentra intenta coincidencia parcial.
    Lanza ValueError si no puede mapear.
    """
    with open(labels_json, "r") as f:
        wnid2name = json.load(f)

    if wnid not in wnid2name:
        raise ValueError(f"WNID {wnid} no está en {labels_json}")

    target_name = wnid2name[wnid].split(",")[0].strip().lower()  # p.ej. "wombat"
    imagenet_classes = weights.meta["categories"]

    # 1) buscar coincidencia exacta (comparando nombres lower)
    for i, cname in enumerate(imagenet_classes):
        if cname.lower().strip() == target_name:
            return i

    # 2) intentar coincidencia parcial por tokens
    target_token = target_name.split()[0]
    for i, cname in enumerate(imagenet_classes):
        if target_token in cname.lower():
            return i

    raise ValueError(f"No pude mapear {wnid} -> índice en weights.meta['categories'] (buscado '{target_name}').")

def global_evaluate(model, metodo, param, VAL_DIR="val.X"):

    model.eval()
    val_dataset = ImageNet100ValDataset(VAL_DIR, transform=transform, labels_json="Labels.json")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for imgs, labels_idx in val_loader:

        # imgs YA están normalizadas (vienen del transform)
        if metodo == None:
            att_imgs = imgs  
        else:
            att_imgs = torch.stack([
                metodo(model, imgs[i], labels_idx[i], param)
                for i in range(len(imgs))
                    ]).to(device)

        with torch.no_grad():
            outputs = model(att_imgs)

            filtered_logits = outputs[:, selected_indices_in_model]
            filtered_probs = F.softmax(filtered_logits, dim=1)

            preds_in_filtered = filtered_probs.argmax(dim=1)
            top5_preds = torch.topk(filtered_probs, 5, dim=1).indices

            pred_wnids = [selected_classes[i] for i in preds_in_filtered]
            true_wnids = [list(val_dataset.class_to_idx.keys())[i] for i in labels_idx]


            for i in range(labels_idx.size(0)):
                if labels_idx[i].item() in top5_preds[i]:
                    correct_top5 += 1

            correct_top1 += sum(p == t for p, t in zip(pred_wnids, true_wnids))
            total += len(imgs)

    return correct_top1/total, correct_top5/total


def global_transfer(model_atk, model_trans, metodo, param, VAL_DIR="val.X"):

    model_atk.eval()
    model_trans.eval()
    val_dataset = ImageNet100ValDataset(VAL_DIR, transform=transform, labels_json="Labels.json")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for imgs, labels_idx in val_loader:

        # imgs YA están normalizadas (vienen del transform)
        if metodo == None:
            att_imgs = imgs  
        else:
            att_imgs = torch.stack([
                metodo(model_atk, imgs[i], labels_idx[i], param)
                for i in range(len(imgs))
                    ]).to(device)

        with torch.no_grad():
            outputs = model_trans(att_imgs)

            filtered_logits = outputs[:, selected_indices_in_model]
            filtered_probs = F.softmax(filtered_logits, dim=1)

            preds_in_filtered = filtered_probs.argmax(dim=1)
            top5_preds = torch.topk(filtered_probs, 5, dim=1).indices

            pred_wnids = [selected_classes[i] for i in preds_in_filtered]
            true_wnids = [list(val_dataset.class_to_idx.keys())[i] for i in labels_idx]


            for i in range(labels_idx.size(0)):
                if labels_idx[i].item() in top5_preds[i]:
                    correct_top5 += 1

            correct_top1 += sum(p == t for p, t in zip(pred_wnids, true_wnids))
            total += len(imgs)

    return correct_top1/total, correct_top5/total

if __name__ == "__main__":

    # --- Ejemplo de uso ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)

    wombat_wnid = "n01883070"   # WNID del wombat (ajusta si tu Labels.json usa otro)
    target_idx = wnid_to_model_index(wombat_wnid, weights, labels_json="Labels.json")
    print("Target index in model outputs:", target_idx)
