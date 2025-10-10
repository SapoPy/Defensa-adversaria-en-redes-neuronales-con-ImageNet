import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import ResNet34_Weights
import json

class ImageNet100ValDataset(Dataset):
    def __init__(self, root_dir, transform=None, labels_json="Labels.json"):
        self.root_dir = root_dir
        self.transform = transform

        # Cargar mapeo global desde Labels.json
        with open(labels_json) as f:
            labels = json.load(f)

        # Usar el orden y mapeo original de índices
        self.class_to_idx = {wnid: i for i, wnid in enumerate(labels.keys())}

        # Cargar las muestras
        self.samples = []
        for wnid in self.class_to_idx:
            class_dir = os.path.join(root_dir, wnid)
            if not os.path.isdir(class_dir):
                continue
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, f), self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

MEAN_DATASET = [0.485, 0.456, 0.406]
STD_DATASET  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(256),                  # Redimensiona el lado más corto a 256 px
    transforms.CenterCrop(224),              # Recorta el centro a 224×224 (tamaño típico de ImageNet)
    transforms.ToTensor(),                   # Convierte a tensor (0–1)
    transforms.Normalize(                    # Normaliza con medias y desv. estándar de ImageNet
        mean= MEAN_DATASET,
        std =STD_DATASET
    )
])

VAL_DIR = "val.X"

with open("Labels.json") as f:
    labels = json.load(f)

selected_classes = list(labels.keys()) 

weights = ResNet34_Weights.DEFAULT

imagenet_classes = weights.meta["categories"]

# Mapear WNID a nombre de clase entendible por el modelo
wnid_to_name = {wnid: labels[wnid].split(',')[0] for wnid in selected_classes}

# Obtener índices dentro de las 1000 clases del modelo
selected_indices_in_model = [imagenet_classes.index(wnid_to_name[wnid]) for wnid in selected_classes]


if __name__ == "__main__":
    val_dataset = ImageNet100ValDataset(VAL_DIR, transform=transform)
    val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    print("Total imágenes de validación:", len(val_dataset))