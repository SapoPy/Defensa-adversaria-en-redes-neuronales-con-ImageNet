from PIL import Image
import io
import torch
import numpy as np
from ataqueFGSM2 import *
from AtaquesGaussianos2 import *
from ImageNet100ValDataset import *
from torchvision.models import resnet34, ResNet34_Weights
from utils import *

def jpeg_defense(img_tensor, quality=90):
    img_tensor = img_tensor.cpu()
    img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))

    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)

    jpeg_img = Image.open(buffer)
    jpeg_img = torch.tensor(np.array(jpeg_img)).permute(2,0,1) / 255.0

    return jpeg_img

def defensaJPEG(ds, modelo, calidad ,parametros, tipo_ataque):

  count  = 0
  for i in range(len(ds)):
      imagen, label = ds[i][0], ds[i][1]
      imagen = imagen.to(device)
      ataque = tipo_ataque(modelo, imagen.to(device), label, parametros)

      defensa = jpeg_defense(denormalize_tensor(ataque).squeeze(0), quality=calidad)
      defensa = defensa.to(device)
      if label == (model(defensa.unsqueeze(0))[:, selected_indices_in_model]).argmax(dim=1):
          count += 1

  return count/len(ds)


def multidefensaJPEG(ds, modelo, calidades ,parametros, tipo_ataque):
  counts = {calidad: 0 for calidad in calidades}
  for i in range(len(ds)):
      imagen, label = ds[i][0], ds[i][1]
      imagen = imagen.to(device)
      ataque = tipo_ataque(modelo, imagen.to(device), label, parametros)
      label = list(ds.class_to_idx.keys())[label]
      for calidad in calidades:
        defensa = jpeg_defense(denormalize_tensor(ataque).squeeze(0), quality=calidad)
        defensa = defensa.to(device)
        pred = (model(defensa.unsqueeze(0))[:, selected_indices_in_model]).argmax(dim=1)
        if label == selected_classes[pred]:
            counts[calidad] += 1
  for calidad in calidades:
    print(f"Para calidad {calidad} JPEG: {counts[calidad]/len(ds)} top 1 acc")

if __name__ == "__main__":
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)

    ds = ImageNet100ValDataset(VAL_DIR, transform)

    count  = 0
    for i in range(len(ds)):
        imagen, label = ds[i][0], ds[i][1]

        ataque = generar_imagen_gaussian(model, imagen, label, [0.05])

        defensa = jpeg_defense(denormalize_tensor(ataque).squeeze(0))

        #print((model(imagen.unsqueeze(0))[:, selected_indices_in_model]).argmax(dim=1))
        #print((model(ataque.unsqueeze(0))[:, selected_indices_in_model]).argmax(dim=1))
        #print((model(defensa.unsqueeze(0))[:, selected_indices_in_model]).argmax(dim=1))
        print(i)
        if (model(imagen.unsqueeze(0))[:, selected_indices_in_model]).argmax(dim=1) == (model(defensa.unsqueeze(0))[:, selected_indices_in_model]).argmax(dim=1):
            count += 1
            print(count)

    print(count/len(ds))