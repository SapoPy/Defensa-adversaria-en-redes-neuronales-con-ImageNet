import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import json

from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
from ImageNet100ValDataset import ImageNet100ValDataset  # tu clase

# -------------------------
# util: cargar modelo y transform recomendado
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights).to(device).eval()
preprocess = weights.transforms()

# -------------------------
# cargar mapping WNID <-> nombre
# -------------------------
def load_wnid_name_map(labels_json="Labels.json"):
    with open(labels_json, "r") as f:
        wnid2name = json.load(f)
    name_to_wnid = {}
    for wnid, longname in wnid2name.items():
        first = longname.split(",")[0].strip().lower()
        name_to_wnid.setdefault(first, []).append(wnid)
    return wnid2name, name_to_wnid

wnid2name, name_to_wnid = load_wnid_name_map("Labels.json")
imagenet_classes = weights.meta["categories"]  # lista de 1000 nombres (ImageNet completo)

# -------------------------
# Seleccionar solo las 100 clases de tu dataset
# -------------------------
dataset_root = Path("val.X")  # carpeta de tu conjunto reducido
selected_classes = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])

# Mapear nombres de clase (wnid) del dataset a índices del modelo
selected_indices_in_model = []
for wnid in selected_classes:
    for i, class_name in enumerate(imagenet_classes):
        if wnid in class_name or wnid.startswith("n"):  # algunos nombres son n01440764
            if wnid in class_name or class_name.startswith(wnid):
                selected_indices_in_model.append(i)
                break

# -------------------------
# top-5 filtrado solo sobre las 100 clases
# -------------------------
def top5_for_image_path(model, img_path, preprocess, imagenet_classes, name_to_wnid, device=device):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)[0]
        # *** Solo usar las 100 clases seleccionadas ***
        logits_filtered = logits[selected_indices_in_model]
        probs = F.softmax(logits_filtered, dim=0)

    topk = torch.topk(probs, k=5)
    indices = topk.indices.cpu().tolist()
    probs_list = topk.values.cpu().tolist()

    results = []
    for idx_filtered, p in zip(indices, probs_list):
        # recuperar índice original (de las 1000)
        idx_model = selected_indices_in_model[idx_filtered]
        class_name = imagenet_classes[idx_model]
        key = class_name.split()[0].lower()
        possible_wnids = name_to_wnid.get(key, [])
        wnid = possible_wnids[0] if possible_wnids else None
        wnid_name = wnid2name[wnid] if wnid else None
        results.append({
            "model_idx": idx_model,
            "model_name": class_name,
            "prob": p,
            "wnid": wnid,
            "wnid_fullname": wnid_name
        })
    return results

# -------------------------
# Desde dataset (ya normalizado)
# -------------------------
def top5_from_dataset_index(model, dataset, idx, preprocess, imagenet_classes, name_to_wnid, device=device):
    img_tensor, _ = dataset[idx]
    if isinstance(img_tensor, torch.Tensor) and img_tensor.dim() == 3:
        x = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0]
            # *** Filtrar a las 100 clases ***
            logits_filtered = logits[selected_indices_in_model]
            probs = F.softmax(logits_filtered, dim=0)

        topk = torch.topk(probs, k=5)
        indices = topk.indices.cpu().tolist()
        probs_list = topk.values.cpu().tolist()

        results = []
        for idx_filtered, p in zip(indices, probs_list):
            idx_model = selected_indices_in_model[idx_filtered]
            class_name = imagenet_classes[idx_model]
            key = class_name.split()[0].lower()
            possible_wnids = name_to_wnid.get(key, [])
            wnid = possible_wnids[0] if possible_wnids else None
            wnid_name = wnid2name[wnid] if wnid else None
            results.append({
                "model_idx": idx_model,
                "model_name": class_name,
                "prob": p,
                "wnid": wnid,
                "wnid_fullname": wnid_name
            })
        return results
    else:
        path = dataset.samples[idx][0]
        return top5_for_image_path(model, path, preprocess, imagenet_classes, name_to_wnid, device=device)

# -------------------------
# Ejemplo de uso
# -------------------------
if __name__ == "__main__":
    img_path = r"val.X\n01440764\ILSVRC2012_val_00000293.JPEG"
    res = top5_for_image_path(model, img_path, preprocess, imagenet_classes, name_to_wnid, device=device)
    print("Top-5 (ruta, solo 100 clases):")
    for r in res:
        print(f"{r['model_idx']:4d}  {r['model_name'][:40]:40s}  p={r['prob']:.4f}  wnid={r['wnid']}  wnid_name={r['wnid_fullname']}")

    ds = ImageNet100ValDataset("val_pgd_targeted", transform=weights.transforms(), labels_json="Labels.json")
    idx = 0
    res2 = top5_from_dataset_index(model, ds, idx, preprocess, imagenet_classes, name_to_wnid, device=device)
    print("\nTop-5 (desde dataset, solo 100 clases):")
    for r in res2:
        print(f"{r['model_idx']:4d}  {r['model_name'][:40]:40s}  p={r['prob']:.4f}  wnid={r['wnid']}  wnid_name={r['wnid_fullname']}")
