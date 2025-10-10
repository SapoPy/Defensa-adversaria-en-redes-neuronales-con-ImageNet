import torch
import os
from torchvision.utils import save_image
from AtaquesFGSM import image_gradient   # tu función existente
from ImageNet100ValDataset import *
from torchvision.models import resnet34, ResNet34_Weights

# Parámetros
picked_classes = ['n01440764', 'n01847000', 'n01883070']  # peces, patos, wombats
VAL_DIR = "3 clases"          # carpeta de validación reducida
OUTPUT_DIR = "FGSM_out"       # carpeta donde se guardarán las perturbaciones
EPSILON = 0.05               # magnitud de perturbación

# Crear carpeta de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar modelo preentrenado
weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)
model.eval()

# Dataset
val_dataset = ImageNet100ValDataset(VAL_DIR, transform=weights.transforms(), labels_json="Labels.json")

# Mean/std (necesarios para pasar de normalizado a pixel-space)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
mean_t = torch.tensor(mean).view(3,1,1)
std_t  = torch.tensor(std).view(3,1,1)

# Iterar sobre cada muestra
print("Generando imágenes FGSM...\n")

for idx in range(len(val_dataset)):
    img_tensor, label_idx = val_dataset[idx]

    # Calcular gradiente
    grad_pixel = image_gradient(model, img_tensor, label_idx,
                                to_pixel_space=True,
                                mean=mean, std=std,
                                )

    # Denormalizar imagen original
    img_pixel = img_tensor * std_t + mean_t
    img_perturbed = img_pixel - EPSILON * grad_pixel
    img_perturbed = torch.clamp(img_perturbed, 0, 1)

    # Guardar resultado
    wnid = list(val_dataset.class_to_idx.keys())[label_idx]
    class_dir = os.path.join(OUTPUT_DIR, wnid)
    os.makedirs(class_dir, exist_ok=True)

    save_path = os.path.join(class_dir, f"img_{idx:05d}_fgsm.png")
    save_image(img_perturbed, save_path)

    if (idx + 1) % 10 == 0 or (idx + 1) == len(val_dataset):
        print(f"  Procesadas {idx + 1}/{len(val_dataset)} imágenes...")

print(f"\nImágenes perturbadas guardadas en: {OUTPUT_DIR}")
