import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
from ImageNet100ValDataset import *
from utils import *   # si necesitas utilidades comunes

# ------------------ Estadísticas de ImageNet ------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ===========================================================
# ✔️ Ataque: ruido Gaussiano en espacio normalizado (misma firma que rFGSM)
# ===========================================================
def generar_imagen_gaussian(model, img_norm, label, args):
    """
    args[0] = sigma (std del ruido Gaussiano) - en espacio NORMALIZADO
    img_norm: imagen NORMALIZADA (C,H,W)
    Devuelve adv_norm: imagen adversarial NORMALIZADA (img_norm + ruido gaussiano)
    """
    sigma = args[0]

    # ruido gaussiano en espacio normalizado
    noise = torch.randn_like(img_norm) * sigma

    adv_norm = img_norm + noise

    # clamp en espacio normalizado (evita valores extremos)
    adv_norm = adv_norm.clamp(-3, 3)

    return adv_norm

# ===========================================================
# ✔️ (Opcional) Guardar dataset atacado con ruido gaussiano
#    - usa la misma convención que guardar_dataset_adv (desnormaliza para PNG)
# ===========================================================
def guardar_dataset_gaussian(model, dataset, sigma, out_dir, device=None, print_every=50):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)

    for i in range(total):
        img_norm, label_idx = dataset[i]

        adv_norm = generar_imagen_gaussian(model, img_norm, label_idx, [sigma])

        # Convertir para guardarlo visualmente:
        adv_pixel = denormalize_tensor(adv_norm.unsqueeze(0)).squeeze(0)

        try:
            orig_path = dataset.samples[i][0]
            wnid = Path(orig_path).parent.name
            fname = Path(orig_path).name
        except:
            wnid = str(label_idx)
            fname = f"img_{i:06d}.png"

        class_dir = out_dir / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        save_path = class_dir / f"{Path(fname).stem}_gauss.png"

        save_image(adv_pixel, str(save_path))

        if (i+1) % print_every == 0 or i+1 == total:
            print(f"[{i+1}/{total}] -> {save_path}")

    return str(out_dir), ImageNet100ValDataset(str(out_dir), transform=dataset.transform)


# ===========================================================
# ✔️ Wrapper de evaluación (misma forma que usas con rFGSM)
#    Llama a global_evaluate tal cual:
#      global_evaluate(model, generar_imagen_gaussian, [sigma], VAL_DIR)
# ===========================================================
def evaluate_gaussian_attack(model, val_dir, sigma):
    return global_evaluate(model, generar_imagen_gaussian, [sigma], val_dir)


# ------------------ ejemplo de uso (estructura igual al rFGSM) ------------------
if __name__ == "__main__":
    from torchvision.models import resnet34, ResNet34_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)

    VAL_DIR = "3 clases"
    val_ds = ImageNet100ValDataset(VAL_DIR, transform=transform)

    sigma = 0.05

    # 1) evaluar on-the-fly con global_evaluate
    acc_top1, acc_top5 = global_evaluate(model, generar_imagen_gaussian, [sigma], VAL_DIR)
    print(f"Precisión top 1 (gaussian on-the-fly): {acc_top1:.4f}")
    print(f"Precisión top 5 (gaussian on-the-fly): {acc_top5:.4f}")
    
    #metodo=None
    #acc_top1, acc_top5 = global_evaluate(model, metodo, [sigma], VAL_DIR)
    #print(f"Precisión top 1 (raw): {acc_top1:.4f}")
    #print(f"Precisión top 5 (raw): {acc_top5:.4f}") 