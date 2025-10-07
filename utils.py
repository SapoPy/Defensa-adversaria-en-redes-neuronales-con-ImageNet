import torch
from torchvision.models import resnet50, ResNet50_Weights
from ImageNet100ValDataset import *

def evaluar_imagen(model, img, selected_indices_in_model):
    """
    Evalúa una sola imagen en el modelo.

    Parámetros:
        model: modelo preentrenado (por ejemplo, resnet50)
        img: tensor de imagen (C, H, W) ya transformado
        selected_indices_in_model: lista de índices de las clases que se quieren evaluar en el modelo

    Devuelve:
        pred_wnid: WNID predicho
        nombre_legible: nombre de la clase predicha
        prob: probabilidad asociada
    """

    input_tensor = img.unsqueeze(0)

    # Inferencia sin gradientes
    with torch.no_grad():
        output = model(input_tensor)

    # Filtrar los logits solo para las clases seleccionadas
    filtered_logits = output[0][selected_indices_in_model]
    filtered_probs = torch.nn.functional.softmax(filtered_logits, dim=0)

    # Elegir la clase más probable
    pred_idx_in_filtered = filtered_probs.argmax().item()
    pred_wnid = selected_classes[pred_idx_in_filtered]
    prob = filtered_probs[pred_idx_in_filtered].item()

    nombre_legible = labels[pred_wnid]

    return pred_wnid, nombre_legible, prob



if __name__ == "__main__":
    # Modelo
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Dataset
    val_dataset = ImageNet100ValDataset(VAL_DIR, transform=weights.transforms())

    imagenet_classes = weights.meta["categories"]
    selected_indices_in_model = [imagenet_classes.index(labels[wnid].split(',')[0])
                                for wnid in selected_classes]


    # Probar con una imagen
    img, wnid_idx = val_dataset[981]
    pred_wnid, nombre_legible, prob = evaluar_imagen(
        model, img, selected_indices_in_model
    )
    print("Etiqueta real:", list(val_dataset.class_to_idx.keys())[wnid_idx])
    print("Predicción:", pred_wnid, prob)