import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from ImageNet100ValDataset import *

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


if __name__ == "__main__":

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
