import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pathlib import Path
from ImageNet100ValDataset import *

# --- Ruido gaussiano ---
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        noisy_tensor = (tensor + noise).clamp_(0., 1.)
        return noisy_tensor, noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# --- Denormalización vectorizada ---
def denormalize(tensor, mean=MEAN_DATASET, std=STD_DATASET):
    mean = torch.tensor(mean, device=tensor.device)[:, None, None]
    std = torch.tensor(std, device=tensor.device)[:, None, None]
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


# --- Aplicar ruido a todo el dataset (sin tqdm, con progreso simple) ---
def apply_noise_to_dataset(dataset_dir="val.X", output_dir="val_noisy", std=0.05):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    noise_fn = AddGaussianNoise(0.0, std)

    # Contar total de imágenes primero
    all_images = [p for p in dataset_dir.rglob("*.JPEG")]
    total_images = len(all_images)

    if total_images == 0:
        print("No se encontraron imágenes en el dataset.")
        return

    print(f"Procesando {total_images} imágenes con ruido gaussiano (std = {std})...")

    for i, img_path in enumerate(all_images, start=1):
        class_rel = img_path.parent.relative_to(dataset_dir)
        class_out = output_dir / class_rel
        class_out.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error al abrir {img_path}: {e}")
            continue

        img_tensor = to_tensor(img)
        noisy_tensor, _ = noise_fn(img_tensor)
        noisy_img = to_pil(noisy_tensor)

        noisy_path = class_out / f"{img_path.stem}_noisy{img_path.suffix}"
        noisy_img.save(noisy_path)

        # Mostrar progreso cada 100 imágenes o al final
        if i % 100 == 0 or i == total_images:
            porcentaje = (i / total_images) * 100
            print(f"Progreso: {i}/{total_images} imágenes ({porcentaje:.1f}%)")

    print(f"\nSe guardaron {total_images} imágenes con ruido gaussiano en {output_dir}")


# --- Mostrar comparación (original / ruido / resultado) ---
def show_noise_example(img_path, std=0.05):
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    noise_fn = AddGaussianNoise(0.0, std)

    img = Image.open(img_path).convert("RGB")
    img_tensor = to_tensor(img)
    noisy_tensor, noise = noise_fn(img_tensor)

    # Normalizar ruido a [0,1] solo para visualizar
    noise_vis = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ruido añadido")
    plt.imshow(noise_vis.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Imagen resultante")
    plt.imshow(noisy_tensor.permute(1, 2, 0))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# --- Ejemplo de uso ---
if __name__ == "__main__":
    # 1) Aplicar ruido a todo el dataset
    apply_noise_to_dataset("val.X", "val_noisy_01", std=0.1)
    apply_noise_to_dataset("val.X", "val_noisy_015", std=0.15)
    # 2) Mostrar ejemplo visual
    # example_path = Path("val.X/n01440764/ILSVRC2012_val_00000293.JPEG")
    # show_noise_example(example_path, std=0.05)
