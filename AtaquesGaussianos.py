import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from ImageNet100ValDataset import *
from pathlib import Path

# --- Ruido gaussiano ---
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Genera y aplica ruido de forma vectorizada
        return (tensor + torch.randn_like(tensor) * self.std + self.mean).clamp_(0., 1.)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# --- Denormalización vectorizada ---
def denormalize(tensor, mean=MEAN_DATASET, std=STD_DATASET):
    """
    Inversa de Normalize() para visualizar imágenes correctamente.
    """
    mean = torch.tensor(mean, device=tensor.device)[:, None, None]
    std = torch.tensor(std, device=tensor.device)[:, None, None]
    return torch.clamp(tensor * std + mean, 0.0, 1.0)

def apply_noise_to_class(wnid, output_dir, dataset_dir = "val.X",std=0.05):
    """
    Aplica ruido gaussiano a todas las imágenes de una clase específica y las guarda.

    Parámetros:
        dataset_dir: Path al dataset (val.X)
        wnid: clase (ej: 'n01440764')
        output_dir: carpeta donde se guardarán las imágenes ruidosas
        std: desviación estándar del ruido
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transform para convertir PIL->Tensor [0,1]
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    noise = AddGaussianNoise(0.0, std)

    # Buscar imágenes de la clase
    class_dir = dataset_dir / wnid
    if not class_dir.exists():
        raise ValueError(f"La clase {wnid} no existe en {dataset_dir}")

    for img_path in class_dir.glob("*.JPEG"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = to_tensor(img)
        noisy_tensor = noise(img_tensor)
        noisy_img = to_pil(noisy_tensor)

        # Guardar con sufijo _noisy
        noisy_path = output_dir / f"{img_path.stem}_noisy{img_path.suffix}"
        noisy_img.save(noisy_path)

    print(f"Se guardaron {len(list(class_dir.glob('*.JPEG')))} imágenes ruidosas en {output_dir}")

# Ejemplo de uso
# apply_noise_to_class("val.X", "n01440764", "val_noisy/n01440764", std=0.05)


if __name__ == "__main__":
    apply_noise_to_class("n01440764", "val_noisy/n01440764")
    # Transformaciones
    pre_norm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    to_model = transforms.Normalize(mean=MEAN_DATASET, std=STD_DATASET)

    # Dataset y carga de imagen
    val_dataset = ImageNet100ValDataset(VAL_DIR, transform=transform)
    img_path = val_dataset.samples[0][0]
    img_pil = Image.open(img_path).convert("RGB")

    # Tensor [0,1]
    img_raw = pre_norm(img_pil)
    noisy_raw = AddGaussianNoise()(img_raw)

    # Normalización para el modelo
    img_for_model = to_model(img_raw)
    noisy_for_model = to_model(noisy_raw)

    # Mostrar imágenes
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img_raw.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Con ruido gaussiano")
    plt.imshow(noisy_raw.permute(1, 2, 0))
    plt.axis("off")

    plt.tight_layout()
    plt.show()
