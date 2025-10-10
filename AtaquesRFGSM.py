import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import resnet34, ResNet34_Weights
from ImageNet100ValDataset import *
from pathlib import Path
from torchvision import transforms
from PIL import Image

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

# --- Ataque RFGSM ---
class RFGSMAttack:
    def __init__(self, epsilon=8/255, alpha=2/255):
        self.epsilon = epsilon
        self.alpha = alpha

    def __call__(self, model, img, label, device=None):
        """
        Ataque RFGSM: Paso aleatorio inicial + FGSM.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Imagen original en pixel space
        img_pixel = denormalize_tensor(img.unsqueeze(0)).squeeze(0)
        
        # Paso aleatorio inicial
        noise = torch.empty_like(img_pixel).uniform_(-self.epsilon, self.epsilon)
        img_random = (img_pixel + noise).clamp(0., 1.)
        
        # Calcular gradiente en el punto aleatorio
        img_random_norm = normalize_for_model(img_random)
        grad = image_gradient(model, img_random_norm, label, to_pixel_space=True, device=device)
        
        # Paso FGSM desde el punto aleatorio
        perturbation = (self.epsilon - self.alpha) * grad.sign()
        img_perturbed = (img_random + perturbation).clamp(0., 1.)
        
        # Proyección final
        delta = img_perturbed - img_pixel
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        img_perturbed = (img_pixel + delta).clamp(0., 1.)
        
        perturbation_final = img_perturbed - img_pixel
        return normalize_for_model(img_perturbed), perturbation_final

def apply_rfgsm_to_all_classes(output_dir, model, dataset_dir="3 clases", epsilon=8/255, alpha=2/255):
    """
    Aplica ataque RFGSM a todas las imágenes de TODAS las clases.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    
    rfgsm = RFGSMAttack(epsilon=epsilon, alpha=alpha)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    total_processed = 0
    
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            wnid = class_dir.name
            output_class_dir = output_dir / wnid
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            class_processed = 0
            for img_path in class_dir.glob("*.JPEG"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = to_tensor(img)
                    img_normalized = normalize(img_tensor)
                    
                    label = 0  # Ajustar según tu dataset
                    
                    adversarial_tensor, _ = rfgsm(model, img_normalized, label, device)
                    
                    adversarial_denorm = denormalize_tensor(adversarial_tensor)
                    adversarial_img = to_pil(adversarial_denorm)
                    
                    adv_path = output_class_dir / f"{img_path.stem}_rfgsm{img_path.suffix}"
                    adversarial_img.save(adv_path)
                    class_processed += 1
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")
            
            print(f"Clase {wnid}: {class_processed} imágenes procesadas")
    
    print(f"Total: {total_processed} imágenes con RFGSM en {output_dir}")

# --- main RFGSM ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)
    
    # Aplicar RFGSM a todas las clases
    apply_rfgsm_to_all_classes("val_rfgsm", model, dataset_dir="3 clases")
    
    # Demo visual con una imagen
    preprocess = weights.transforms()
    val_ds = ImageNet100ValDataset(VAL_DIR, transform=preprocess)
    img_tensor, label_idx = val_ds[0]

    # Aplicar RFGSM
    rfgsm_attack = RFGSMAttack(epsilon=8/255, alpha=2/255)
    img_rfgsm, perturbation = rfgsm_attack(model, img_tensor, label_idx, device)

    # 1) Visualizaciones
    img_pixel = denormalize_tensor(img_tensor.unsqueeze(0)).squeeze(0)
    img_rfgsm_pixel = denormalize_tensor(img_rfgsm.unsqueeze(0)).squeeze(0)
    grad_np = perturbation.permute(1,2,0).detach().numpy()

    # Mostrar canales
    plt.figure(figsize=(12,4))
    for i, ch in enumerate(['R','G','B']):
        plt.subplot(1,3,i+1)
        chan = grad_np[:,:,i]
        norm = (chan - chan.min()) / (chan.max() - chan.min() + 1e-8)
        plt.imshow(norm, cmap='gray')
        plt.title(f'Perturbación RFGSM {ch}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Mostrar mapa combinado (L2)
    grad_comb = torch.norm(perturbation, dim=0).detach().numpy()
    grad_comb_norm = (grad_comb - grad_comb.min()) / (grad_comb.max() - grad_comb.min() + 1e-8)
    plt.figure(figsize=(5,5))
    plt.imshow(grad_comb_norm, cmap='viridis')
    plt.title('Perturbación RFGSM L2 combinada')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    # mostrar original vs perturbada
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].imshow(img_pixel.permute(1,2,0).detach().numpy()); axs[0].set_title('Original'); axs[0].axis('off')
    axs[1].imshow(img_rfgsm_pixel.permute(1,2,0).detach().numpy()); axs[1].set_title('RFGSM Attack'); axs[1].axis('off')
    plt.show()
    
    # Evaluación
    import utils as utils

    imagenet_classes = weights.meta["categories"]
    selected_indices_in_model = [imagenet_classes.index(labels[wnid].split(',')[0])
                                for wnid in selected_classes]
    model.eval()
    print(f"RFGSM Attack with eps=8/255, alpha=2/255:")
    pred_wnid, nombre_legible, prob = utils.evaluar_imagen(model, img_tensor, selected_indices_in_model)
    print(f"Prediction on original image:")
    print(pred_wnid, nombre_legible, prob)
    pred_wnid, nombre_legible, prob = utils.evaluar_imagen(model, img_rfgsm, selected_indices_in_model)
    print(f"Prediction on RFGSM attacked image:")
    print(pred_wnid, nombre_legible, prob)
