import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from ImageNet100ValDataset import *
import json

def evaluate_resnet18():
    # Modelo preentrenado ResNet18
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    # Transformación definida en el script ImageClass
    preprocess = transform

    # Cargar los 100 labels
    with open("Labels.json") as f:
        labels = json.load(f)

    selected_classes = list(labels.keys())
    wnid_to_idx = {wnid: i for i, wnid in enumerate(selected_classes)}

    # Dataset
    val_dataset = ImageNet100ValDataset(VAL_DIR, transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    imagenet_classes = weights.meta["categories"]
    selected_indices_in_model = [
        imagenet_classes.index(labels[wnid].split(',')[0])
        for wnid in selected_classes
    ]

    # Evaluación sobre todas las imágenes
    correct = 0
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

            # Mapeo de índice filtrado → wnid real
            pred_wnids = [selected_classes[i] for i in preds_in_filtered]
            true_wnids = [list(val_dataset.class_to_idx.keys())[i] for i in labels_idx]

            # Calcular precisión
            correct += sum(p == t for p, t in zip(pred_wnids, true_wnids))
            total += len(imgs)

    acc = correct / total
    return acc
    
if __name__ == "__main__":
    print(f"Precisión en validación: {evaluate_resnet18():.4f}")
