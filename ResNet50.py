import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
from ImageClass import *
import json

# Modelo preentrenado ResNet50
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Transformación definida en el script ImageClass
preprocess = transform

# Cargar los 100 labels
with open("Labels.json") as f:
    labels = json.load(f)
selected_classes = list(labels.keys()) # Tienen la forma n01440764
wnid_to_idx = {wnid: i for i, wnid in enumerate(selected_classes)}  # Diccionario con la clase 
                                                                    # (del tipo n01440764) y con indice

# Dataset
val_dataset = ImageNet100ValDataset(VAL_DIR, transform=preprocess)

img, wnid = val_dataset[51]  # Imagen de ejemplo
img_wnid = list(val_dataset.class_to_idx.keys())[wnid]  # obtener WNID real
print("WNID de la imagen:", img_wnid)

# evalua la imagen en el modelo
input_tensor = img.unsqueeze(0) 
with torch.no_grad():
    output = model(input_tensor)  # logits de todas las 1000 clases

imagenet_classes = weights.meta["categories"]  # lista 1000 nombres de ImageNet
selected_indices_in_model = [imagenet_classes.index(labels[wnid].split(',')[0])
                             for wnid in selected_classes]  # se toma solo la primera palabra del json

filtered_logits = output[0][selected_indices_in_model] # seleccionamos los logit de las clases del conjunto de valdiacion
filtered_probs = torch.nn.functional.softmax(filtered_logits, dim=0) # aplica softmax en las 100 clases

pred_idx_in_filtered = filtered_probs.argmax().item()
pred_wnid = selected_classes[pred_idx_in_filtered]
print("Clase predicha (WNID):", pred_wnid)
print("Nombre legible:", labels[pred_wnid])
print("Probabilidad:", filtered_probs[pred_idx_in_filtered].item())
