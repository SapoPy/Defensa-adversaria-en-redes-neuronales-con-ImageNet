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
preprocess = weights.transforms()  # transforma apropiada (normaliza)

# -------------------------
# util: cargar mapping WNID <- -> nombre (Labels.json)
# -------------------------
def load_wnid_name_map(labels_json="Labels.json"):
    with open(labels_json, "r") as f:
        wnid2name = json.load(f)         # wnid -> "nombre, alias..."
    # crear reverse map aproximado name_first_token.lower() -> wnid
    name_to_wnid = {}
    for wnid, longname in wnid2name.items():
        first = longname.split(",")[0].strip().lower()
        name_to_wnid.setdefault(first, []).append(wnid)
    return wnid2name, name_to_wnid

wnid2name, name_to_wnid = load_wnid_name_map("Labels.json")
imagenet_classes = weights.meta["categories"]  # lista 1000 nombres (strings)

# -------------------------
# función principal: top-5 para una imagen (ruta)
# -------------------------
def top5_for_image_path(model, img_path, preprocess, imagenet_classes, name_to_wnid, device=device):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)   # (1,C,H,W)
    with torch.no_grad():
        logits = model(x)[0]                       # (1000,)
        probs = F.softmax(logits, dim=0)
    topk = torch.topk(probs, k=5)
    indices = topk.indices.cpu().tolist()
    probs_list = topk.values.cpu().tolist()

    results = []
    for idx, p in zip(indices, probs_list):
        class_name = imagenet_classes[idx]
        # intentar mapear class_name -> wnid (por primera palabra)
        key = class_name.split()[0].lower()
        possible_wnids = name_to_wnid.get(key, [])
        wnid = possible_wnids[0] if possible_wnids else None
        wnid_name = wnid2name[wnid] if wnid else None
        results.append({"model_idx": idx, "model_name": class_name, "prob": p, "wnid": wnid, "wnid_fullname": wnid_name})
    return results

# -------------------------
# ayuda: obtener imagen desde ImageNet100ValDataset por índice
# -------------------------
def top5_from_dataset_index(model, dataset, idx, preprocess, imagenet_classes, name_to_wnid, device=device):
    # Si dataset ya aplica transform (normalizado) usamos su tensor directamente:
    img_tensor, _ = dataset[idx]  # img_tensor debería estar normalizado si dataset.transform aplica Normalize
    # Si dataset.transform == preprocess entonces img_tensor ya normalizado; sino denormaliza/normaliza según convenga.
    if isinstance(img_tensor, torch.Tensor) and img_tensor.dim()==3:
        x = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0]
            probs = F.softmax(logits, dim=0)
        topk = torch.topk(probs, k=5)
        indices = topk.indices.cpu().tolist()
        probs_list = topk.values.cpu().tolist()
        results = []
        for idx_model, p in zip(indices, probs_list):
            class_name = imagenet_classes[idx_model]
            key = class_name.split()[0].lower()
            possible_wnids = name_to_wnid.get(key, [])
            wnid = possible_wnids[0] if possible_wnids else None
            wnid_name = wnid2name[wnid] if wnid else None
            results.append({"model_idx": idx_model, "model_name": class_name, "prob": p, "wnid": wnid, "wnid_fullname": wnid_name})
        return results
    else:
        # fallback: convertir desde ruta si dataset guarda .samples
        try:
            path = dataset.samples[idx][0]
            return top5_for_image_path(model, path, preprocess, imagenet_classes, name_to_wnid, device=device)
        except Exception:
            raise RuntimeError("Dataset no proporciona tensor ni samples; pasa una ruta o adapta el código.")

# -------------------------
# Ejemplo de uso (main)
# -------------------------
if __name__ == "__main__":
    # 1) evaluar una imagen suelta (ruta)
    img_path = r"val_pgd_targeted\n01440764\ILSVRC2012_val_00000293_adv.png"   # cambia por una ruta real
    res = top5_for_image_path(model, img_path, preprocess, imagenet_classes, name_to_wnid, device=device)
    print("Top-5 (ruta):")
    for r in res:
        print(f"{r['model_idx']:4d}  {r['model_name'][:40]:40s}  p={r['prob']:.4f}  wnid={r['wnid']}  wnid_name={r['wnid_fullname']}")

    # 2) evaluar una muestra desde ImageNet100ValDataset
    ds = ImageNet100ValDataset("val_pgd_targeted", transform=weights.transforms(), labels_json="Labels.json")
    idx = 0
    res2 = top5_from_dataset_index(model, ds, idx, preprocess, imagenet_classes, name_to_wnid, device=device)
    print("\nTop-5 (desde dataset):")
    for r in res2:
        print(f"{r['model_idx']:4d}  {r['model_name'][:40]:40s}  p={r['prob']:.4f}  wnid={r['wnid']}  wnid_name={r['wnid_fullname']}")
