import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from ImageNet100ValDataset import *

def evaluate_resnet34(VAL_DIR="val.X"):
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)
    model.eval()

    val_dataset = ImageNet100ValDataset(VAL_DIR, transform=transform, labels_json="Labels.json")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    imagenet_classes = weights.meta["categories"]

    # Mapear WNID a nombre de clase entendible por el modelo
    wnid_to_name = {wnid: labels[wnid].split(',')[0] for wnid in selected_classes}

    # Obtener índices dentro de las 1000 clases del modelo
    selected_indices_in_model = [imagenet_classes.index(wnid_to_name[wnid]) for wnid in selected_classes]


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
            true_wnids = [list(val_dataset.class_to_idx.keys())[i] for i in labels_idx]
            
            # Sumar aciertos
            correct_top5 += sum(p in t for p, t in zip(preds_in_filtered, top5_preds))
            correct_top1 += sum(p == t for p, t in zip(pred_wnids, true_wnids))
            total += len(imgs)

    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total

    return acc_top1, acc_top5
    
if __name__ == "__main__":
    acc_top1, acc_top5 = evaluate_resnet34()
    print(f"Precisión top 1 en validación: {acc_top1:.4f}")
    print(f"Precisión top 5 en validación: {acc_top5:.4f}")
