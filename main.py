import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import json
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
from ImageNet100ValDataset import ImageNet100ValDataset, transform

# -------------------------
# Configuración base
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights).to(device).eval()
imagenet_classes = weights.meta["categories"]

# -------------------------
# Cargar mapping WNID <-> nombre
# -------------------------
with open("Labels.json", "r") as f:
    wnid2name = json.load(f)

# Carpetas del dataset (100 clases)
dataset_root = Path("val.X")
selected_classes = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])

# -------------------------
# Mapeo robusto WNID → índice de clase del modelo
# -------------------------
wnid_to_model_idx = {}
for wnid in selected_classes:
    if wnid not in wnid2name:
        continue
    wnid_name = wnid2name[wnid].split(",")[0].lower().strip()
    # Buscar coincidencia dentro de las 1000 clases de ImageNet
    match = [i for i, c in enumerate(imagenet_classes) if wnid_name in c.lower()]
    if match:
        wnid_to_model_idx[wnid] = match[0]

selected_indices_in_model = list(wnid_to_model_idx.values())
model_idx_to_wnid = {v: k for k, v in wnid_to_model_idx.items()}

print(f"Total carpetas detectadas en val.X: {len(selected_classes)}")
print(f"Total clases mapeadas correctamente: {len(selected_indices_in_model)}")

if not selected_indices_in_model:
    raise ValueError("No se encontró ninguna clase de val.X en las 1000 del modelo. Verifica Labels.json y los nombres de carpeta.")

# -------------------------
# Función Top-5 (solo tus 100 clases)
# -------------------------
def top5_for_image_path(model, img_path, preprocess, device=device):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)[0]
        logits_filtered = logits[selected_indices_in_model]
        probs = F.softmax(logits_filtered, dim=0)

    k = min(5, len(selected_indices_in_model))
    topk = torch.topk(probs, k=k)
    results = []
    for idx_f, p in zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist()):
        model_idx = selected_indices_in_model[idx_f]
        wnid = model_idx_to_wnid[model_idx]
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

# -------------------------
# Ejemplo de uso
# -------------------------
if __name__ == "__main__":
    img_path = r"val.X\n01440764\ILSVRC2012_val_00000293.JPEG"

    res = top5_for_image_path(model, img_path, transform, device=device)
    print("Top-5 imagen pura:")
    for r in res:
        print(f"{r['model_idx']:4d}  {r['class_name'][:35]:35s}  p={r['prob']:.4f}  wnid={r['wnid']}  wnid_name={r['wnid_name']}")

    img_path = r"val_noisy\n01440764\ILSVRC2012_val_00000293_noisy.jpeg"
    
    res = top5_for_image_path(model, img_path, transform, device=device)
    print("\nTop-5 imagen por ruido blanco:")
    for r in res:
        print(f"{r['model_idx']:4d}  {r['class_name'][:35]:35s}  p={r['prob']:.4f}  wnid={r['wnid']}  wnid_name={r['wnid_name']}")

    img_path = r"FGSM_out\n01440764\img_00050_fgsm.png"
    res = top5_for_image_path(model, img_path, transform, device=device)
    print("\nTop-5 imagen por FGSM:")
    for r in res:
        print(f"{r['model_idx']:4d}  {r['class_name'][:35]:35s}  p={r['prob']:.4f}  wnid={r['wnid']}  wnid_name={r['wnid_name']}")

    img_path = r"RFGSM_out\n01440764\img_00050_rfgsm.png"
    res = top5_for_image_path(model, img_path, transform, device=device)
    print("\nTop-5 imagen por RFGSM:")
    for r in res:
        print(f"{r['model_idx']:4d}  {r['class_name'][:35]:35s}  p={r['prob']:.4f}  wnid={r['wnid']}  wnid_name={r['wnid_name']}")

    img_path = r"val_pgd_targeted\n01440764\ILSVRC2012_val_00000293_adv.png"
    
    res = top5_for_image_path(model, img_path, transform, device=device)
    print("\nTop-5 imagen por PDG:")
    for r in res:
        print(f"{r['model_idx']:4d}  {r['class_name'][:35]:35s}  p={r['prob']:.4f}  wnid={r['wnid']}  wnid_name={r['wnid_name']}")