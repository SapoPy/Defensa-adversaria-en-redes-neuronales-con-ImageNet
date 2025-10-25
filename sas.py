import torch
from torchvision import models
from torch.utils.data import DataLoader
from ImageNet100ValDataset import *


# --- Parámetros ---
target_class_idx = 106  # índice de la clase que quieres contar
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def contador_dirigido(model, val_loader, target_class_idx):
    # --- Contador ---
    count_top1_target = 0
    count_top5_target = 0
    total_images = 0

    with torch.no_grad():
        for imgs, labels_idx in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)

            filtered_logits = outputs[:, selected_indices_in_model]
            filtered_probs = torch.nn.functional.softmax(filtered_logits, dim=1)
            
            preds_top1 = outputs.argmax(dim=1)  # clase con mayor probabilidad
            top5_preds = torch.topk(filtered_probs, 5, dim=1).indices

            for i in range(labels_idx.size(0)):
                if target_class_idx in top5_preds[i]:
                    count_top5_target += 1

            count_top1_target += (preds_top1 == target_class_idx).sum().item()
            total_images += imgs.size(0)
    return count_top1_target/total_images, count_top5_target/total_images

# --- Modelo (ejemplo con ResNet34 preentrenado) ---
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval()

# --- Dataset (ejemplo con ImageNet o carpeta similar) ---
carpeta = "FGSM_targeted_n01883070"
transform = models.ResNet34_Weights.DEFAULT.transforms()
val_ds = ImageNet100ValDataset(carpeta, transform=transform)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


count_top1_target, count_top5_target = contador_dirigido(model, val_loader, target_class_idx)
print(f"Modelo ResNet34")
print(f"Dirigidos a Wombat por FGSM")
print(f"Porcentaje Top-1: {100 * count_top1_target:.2f}%")
print(f"Porcentaje Top-5: {100 * count_top5_target:.2f}%")
print(f"\nDirigido a Wombat por RFGSM")


carpeta = "RFGSM_dirigido_target_106"
val_ds = ImageNet100ValDataset(carpeta, transform=transform)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

count_top1_target, count_top5_target = contador_dirigido(model, val_loader, target_class_idx)

print(f"Porcentaje Top-1: {100 * count_top1_target:.2f}%")
print(f"Porcentaje Top-5: {100 * count_top5_target:.2f}%")
print(f"")

carpeta = "PGD_dirigido_target_106_zip (1)"
val_ds = ImageNet100ValDataset(carpeta, transform=transform)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

count_top1_target, count_top5_target = contador_dirigido(model, val_loader, target_class_idx)

print(f"\nDirigidos a Wombat por PGD")
print(f"Porcentaje Top-1: {100 * count_top1_target:.2f}%")
print(f"Porcentaje Top-5: {100 * count_top5_target:.2f}%")
print(f"")


# --- Modelo (ejemplo con ResNet34 preentrenado) ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()

carpeta = "FGSM_targeted_n01883070"
val_ds = ImageNet100ValDataset(carpeta, transform=transform)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


count_top1_target, count_top5_target = contador_dirigido(model, val_loader, target_class_idx)

print(f"Modelo ResNet50")
print(f"Dirigidos a Wombat por FGSM")
print(f"Porcentaje Top-1: {100 * count_top1_target:.2f}%")
print(f"Porcentaje Top-5: {100 * count_top5_target:.2f}%")
print(f"\nDirigido a Wombat por RFGSM")


carpeta = "RFGSM_dirigido_target_106"
val_ds = ImageNet100ValDataset(carpeta, transform=transform)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)




count_top1_target, count_top5_target = contador_dirigido(model, val_loader, target_class_idx)

print(f"Porcentaje Top-1: {100 * count_top1_target:.2f}%")
print(f"Porcentaje Top-5: {100 * count_top5_target:.2f}%")
print(f"")

print(f"\nDirigido a Wombat por PGD")
carpeta = "PGD_dirigido_target_106_zip (1)"
val_ds = ImageNet100ValDataset(carpeta, transform=transform)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

count_top1_target, count_top5_target = contador_dirigido(model, val_loader, target_class_idx)

print(f"Porcentaje Top-1: {100 * count_top1_target:.2f}%")
print(f"Porcentaje Top-5: {100 * count_top5_target:.2f}%")
print(f"")