import torch
import matplotlib.pyplot as plt
from torchvision.models import resnet34, ResNet34_Weights
from ImageNet100ValDataset import *
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# --- constantes ImageNet ---
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# --- utilidades ---
def denormalize_tensor(tensor, mean=MEAN, std=STD):
    """tensor (C,H,W) o (B,C,H,W) normalizado -> pixel-space [0,1]"""
    m = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1) if tensor.dim()==4 else torch.tensor(mean, device=tensor.device).view(-1,1,1)
    s = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1) if tensor.dim()==4 else torch.tensor(std, device=tensor.device).view(-1,1,1)
    return (tensor * s + m).clamp(0.0, 1.0)

def normalize_for_model(tensor, mean=MEAN, std=STD):
    m = torch.tensor(mean, device=tensor.device).view(-1,1,1)
    s = torch.tensor(std, device=tensor.device).view(-1,1,1)
    return (tensor - m) / s

# --- función para obtener dL/dx ---
def image_gradient(model, img, label,
                   loss_fn=F.cross_entropy,
                   device=None,
                   to_pixel_space=False,
                   mean=MEAN, std=STD):
    """
    img: tensor (C,H,W) O (1,C,H,W) ya normalizado si lo vas a pasar directo al modelo.
    label: int o tensor escalar.
    to_pixel_space: si True devuelve grad en pixel-space (mismo dominio que imagen en [0,1]).
    Devuelve grad tensor (misma forma que img: si pasaste (C,H,W) devuelve (C,H,W)).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    was_training = model.training

    # asegurar batch dim
    squeeze_out = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze_out = True

    img = img.to(device).detach()
    img.requires_grad_(True)

    model.eval()
    model.zero_grad()

    label_t = torch.tensor(label, device=device) if not torch.is_tensor(label) else label.to(device)
    if label_t.dim() == 0:
        label_t = label_t.unsqueeze(0)

    out = model(img)
    loss = loss_fn(out, label_t)
    loss.backward()

    grad = img.grad.detach()  # (1,C,H,W)

    if to_pixel_space:
        # dL/dx_pixel = dL/dx_norm * (1/std)
        std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
        grad = grad / std_t

    if squeeze_out:
        grad = grad.squeeze(0)   # (C,H,W)

    if was_training:
        model.train()

    return grad.cpu()

# --- Ataque Sal y Pimienta ---
class SaltPepperAttack:
    def __init__(self, noise_prob=0.05):
        self.noise_prob = noise_prob

    def __call__(self, img):
        """
        Aplica ruido sal y pimienta a la imagen.
        img: tensor en pixel space [0,1] (C,H,W)
        """
        noisy_img = img.clone()
        h, w = img.shape[1], img.shape[2]
        
        # Máscara para sal (blanco)
        salt_mask = torch.rand(h, w) < self.noise_prob/2
        # Máscara para pimienta (negro)  
        pepper_mask = torch.rand(h, w) < self.noise_prob/2
        
        # Aplicar a cada canal
        for c in range(3):
            noisy_img[c, salt_mask] = 1.0    # Sal (blanco)
            noisy_img[c, pepper_mask] = 0.0  # Pimienta (negro)
            
        return noisy_img

def apply_salt_pepper_to_all_classes(output_dir, dataset_dir="3 clases", noise_prob=0.05):
    """
    Aplica ruido sal y pimienta a todas las imágenes de TODAS las clases y las guarda manteniendo la estructura de carpetas.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Transformaciones
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    # Inicializar ataque
    sp_attack = SaltPepperAttack(noise_prob=noise_prob)
    
    total_processed = 0
    
    # Procesar cada clase en el dataset
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            wnid = class_dir.name
            output_class_dir = output_dir / wnid
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            class_processed = 0
            for img_path in class_dir.glob("*.JPEG"):
                try:
                    # Cargar imagen
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = to_tensor(img)  # Ya está en [0,1]
                    
                    # Aplicar sal y pimienta
                    img_sp = sp_attack(img_tensor)
                    img_sp_pil = to_pil(img_sp)
                    
                    # Guardar con sufijo _sp 
                    sp_path = output_class_dir / f"{img_path.stem}_sp{img_path.suffix}"
                    img_sp_pil.save(sp_path)
                    class_processed += 1
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")
            
            print(f"Clase {wnid}: {class_processed} imágenes procesadas")
    
    print(f"Total: {total_processed} imágenes con Salt & Pepper en {output_dir}")

# --- main Sal y Pimienta ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)
    
    # Aplicar a TODAS las clases
    apply_salt_pepper_to_all_classes("val_sp", dataset_dir="3 clases", noise_prob=0.05)
    
    # Demo visual con evaluación del efecto
    preprocess = weights.transforms()
    
    # Cargar dataset para evaluación
    val_ds = ImageNet100ValDataset(VAL_DIR, transform=preprocess)
    img_tensor, label_idx = val_ds[0]
    
    # Aplicar ataque para visualización
    sp_attack = SaltPepperAttack(noise_prob=0.05)
    img_pixel = denormalize_tensor(img_tensor.unsqueeze(0)).squeeze(0)
    img_sp = sp_attack(img_pixel)
    
    # Normalizar la imagen atacada para el modelo
    img_sp_norm = normalize_for_model(img_sp)

    # 1) VISUALIZACIÓN DE LA PERTURBACIÓN 
    # Calcular la diferencia 
    perturbacion = (img_sp - img_pixel).abs()
    perturbacion_np = perturbacion.permute(1, 2, 0).detach().numpy()

    # Mostrar canales de perturbación
    plt.figure(figsize=(12, 4))
    for i, ch in enumerate(['R', 'G', 'B']):
        plt.subplot(1, 3, i + 1)
        chan = perturbacion_np[:, :, i]
        if chan.max() > 0:
            norm = (chan - chan.min()) / (chan.max() - chan.min() + 1e-8)
        else:
            norm = chan
        plt.imshow(norm, cmap='gray')
        plt.title(f'Perturbación SP {ch}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 2) MAPA COMBINADO L2 DE PERTURBACIÓN
    perturbacion_comb = torch.norm(perturbacion, dim=0).detach().numpy()
    if perturbacion_comb.max() > 0:
        perturbacion_comb_norm = (perturbacion_comb - perturbacion_comb.min()) / (perturbacion_comb.max() - perturbacion_comb.min() + 1e-8)
    else:
        perturbacion_comb_norm = perturbacion_comb
        
    plt.figure(figsize=(5, 5))
    plt.imshow(perturbacion_comb_norm, cmap='viridis')
    plt.title('Perturbación SP L2 combinada')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    # 3) COMPARACIÓN ORIGINAL VS ATACADA
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_pixel.permute(1, 2, 0).detach().numpy())
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    axs[1].imshow(img_sp.permute(1, 2, 0).detach().numpy())
    axs[1].set_title('Salt & Pepper Attack')
    axs[1].axis('off')
    plt.show()

    # 4) EVALUACIÓN DEL EFECTO EN EL MODELO
    import utils as utils
    
    # Obtener clases de ImageNet
    imagenet_classes = weights.meta["categories"]
    
    # Obtener los índices de las clases seleccionadas en el modelo
    selected_indices_in_model = [imagenet_classes.index(labels[wnid].split(',')[0])
                                for wnid in selected_classes]
    
    model.eval()
    
    print(f"Salt & Pepper Attack with noise_prob={0.05}:")
    
    # Evaluar imagen original
    pred_wnid_orig, nombre_legible_orig, prob_orig = utils.evaluar_imagen(
        model, img_tensor, selected_indices_in_model
    )
    print(f"Prediction on original image:")
    print(f"{pred_wnid_orig}, {nombre_legible_orig}, {prob_orig}")
    
    # Evaluar imagen atacada
    pred_wnid_sp, nombre_legible_sp, prob_sp = utils.evaluar_imagen(
        model, img_sp_norm, selected_indices_in_model
    )
    print(f"Prediction on Salt & Pepper attacked image:")
    print(f"{pred_wnid_sp}, {nombre_legible_sp}, {prob_sp}")
    
    # Mostrar si el ataque fue exitoso
    if pred_wnid_orig != pred_wnid_sp:
        print("ATAQUE EXITOSO: La predicción cambió!")
    else:
        print("Ataque fallido: La predicción se mantuvo igual")
    
    # Mostrar diferencia en probabilidad
    diff_prob = prob_orig - prob_sp
    print(f"Diferencia en probabilidad: {diff_prob:.4f}")
