import torch
import os
from torchvision.utils import save_image
from AtaquesFGSM import image_gradient   # tu función existente
from ImageNet100ValDataset import *
from torchvision.models import resnet34, ResNet34_Weights
from AtaquesGaussianos import *

# --- Parámetros ---
picked_classes = ['n01440764', 'n01847000', 'n01883070']  # peces, patos, wombats
VAL_DIR = "3 clases"
OUTPUT_DIR = "RFGSM_out"
EPSILON = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)
model.eval()

val_dataset = ImageNet100ValDataset(
    VAL_DIR,
    transform=weights.transforms(),
    labels_json="Labels.json"
)

mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

# --- Iterar sobre cada imagen ---
print("Generando imágenes R-FGSM con inicio gaussiano...\n")
noise_adder = AddGaussianNoise(0.0, EPSILON)  # ruido inicial en pixel-space

for idx in range(len(val_dataset)):
    img_norm, label_idx = val_dataset[idx]       # img ya normalizado
    img_pixel = img_norm * std + mean            # denormalizar a [0,1]

    # --- Paso 1: inicio aleatorio usando GaussianNoise ---
    img_rand_pixel = torch.clamp(noise_adder(img_pixel), 0.0, 1.0)

    # --- Paso 2: convertir a normalizado para el modelo ---
    img_rand_norm = (img_rand_pixel - mean) / std

    # --- Paso 3: calcular gradiente (espacio normalizado) ---
    grad_norm = image_gradient(model, img_rand_norm, label_idx,
                               to_pixel_space=False,
                               mean=[0.485,0.456,0.406],
                               std=[0.229,0.224,0.225])

    # --- Paso 4: FGSM step ---
    eps_norm = (EPSILON / std).view(3,1,1)
    adv_norm = img_rand_norm + eps_norm * grad_norm.sign()
    # clamp en espacio normalizado
    min_norm = ((0.0 - mean)/std)
    max_norm = ((1.0 - mean)/std)
    adv_norm = torch.max(torch.min(adv_norm, max_norm), min_norm)

    # --- Paso 5: denormalizar para guardar ---
    adv_pixel = (adv_norm * std + mean).clamp(0.0, 1.0)

    wnid = list(val_dataset.class_to_idx.keys())[label_idx]
    class_dir = os.path.join(OUTPUT_DIR, wnid)
    os.makedirs(class_dir, exist_ok=True)

    save_path = os.path.join(class_dir, f"img_{idx:05d}_rfgsm.png")
    save_image(adv_pixel, save_path)

    if (idx+1) % 10 == 0 or (idx+1) == len(val_dataset):
        print(f"  Procesadas {idx+1}/{len(val_dataset)} imágenes...")

print(f"\nImágenes R-FGSM guardadas en: {OUTPUT_DIR}")