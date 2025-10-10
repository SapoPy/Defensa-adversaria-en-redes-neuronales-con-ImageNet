import torch
from torch.utils.data import DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from ImageNet100ValDataset import *

def evaluate_densenet121(VAL_DIR="val.X"):
    # Modelo preentrenado densenet121
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)
    model.eval()

    val_dataset = ImageNet100ValDataset(VAL_DIR, transform=transform, labels_json="Labels.json")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
            for i in range(labels_idx.size(0)):
                if labels_idx[i].item() in top5_preds[i]:
                    correct_top5 += 1
                    
            correct_top1 += sum(p == t for p, t in zip(pred_wnids, true_wnids))
            total += len(imgs)

    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total

    return acc_top1, acc_top5
    
if __name__ == "__main__":
    acc_top1, acc_top5 = evaluate_densenet121()
    print(f"Precisión top 1 en validación: {acc_top1:.4f}")
    print(f"Precisión top 5 en validación: {acc_top5:.4f}")
