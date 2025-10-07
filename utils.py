import torch
from torchvision.models import resnet50, ResNet50_Weights
from ImageNet100ValDataset import *
import json

def evaluar_imagen(model, img):
    """
    Evalúa una sola imagen en el modelo.

    Parámetros:
        model: modelo preentrenado 
        img: tensor de imagen (C, H, W) ya transformado
        wnid_real: opcional, el WNID real de la imagen

    Devuelve:
        pred_wnid: WNID predicho
        nombre_legible: nombre de clase
        probabilidad: valor de probabilidad
    """
    input_tensor = img.unsqueeze(0)  # agrega batch dimension

    with torch.no_grad():
        output = model(input_tensor)

    # filtrar solo tus 100 clases
    filtered_logits = output[0][selected_indices_in_model]
    filtered_probs = torch.nn.functional.softmax(filtered_logits, dim=0)

    pred_idx_in_filtered = filtered_probs.argmax().item()
    pred_wnid = selected_classes[pred_idx_in_filtered]
    prob = filtered_probs[pred_idx_in_filtered].item()

    nombre_legible = labels[pred_wnid]

    return pred_wnid, nombre_legible, prob


if __name__ == "__main__":
    # Modelo para ejemplo
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    preprocess = transform  # definida en ImageNet100ValDataset

    with open("Labels.json") as f:
        labels = json.load(f)

    selected_classes = list(labels.keys()) 
    imagenet_classes = weights.meta["categories"]

    # índices de las 100 clases dentro del modelo
    selected_indices_in_model = [
        imagenet_classes.index(labels[wnid].split(',')[0])
        for wnid in selected_classes
    ]



    val_dataset = ImageNet100ValDataset(VAL_DIR, transform=preprocess)

    img, wnid_idx = val_dataset[51]
    wnid_real = list(val_dataset.class_to_idx.keys())[wnid_idx]

    pred_wnid, nombre, prob = evaluar_imagen(model, img)

    print(f"Clase real: {wnid_real}")
    print(f"Predicha:   {pred_wnid}")
    print(f"Nombre:     {nombre}")
    print(f"Probabilidad: {prob:.4f}")