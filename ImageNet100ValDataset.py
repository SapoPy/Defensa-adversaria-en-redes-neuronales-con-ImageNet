import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageNet100ValDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(sorted(os.listdir(root_dir)))}
        self.samples = [
            (os.path.join(root_dir, c, f), self.class_to_idx[c])
            for c in self.class_to_idx
            for f in os.listdir(os.path.join(root_dir, c))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


transform = transforms.Compose([
    transforms.Resize(256),                  # Redimensiona el lado más corto a 256 px
    transforms.CenterCrop(224),              # Recorta el centro a 224×224 (tamaño típico de ImageNet)
    transforms.ToTensor(),                   # Convierte a tensor (0–1)
    transforms.Normalize(                    # Normaliza con medias y desv. estándar de ImageNet
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

VAL_DIR = "val.X"

if __name__ == "__main__":
    val_dataset = ImageNet100ValDataset(VAL_DIR, transform=transform)
    val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    print("Total imágenes de validación:", len(val_dataset))