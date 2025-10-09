import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import resnet34, ResNet34_Weights
from ImageNet100ValDataset import *

# --- constantes ImageNet ---
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# --- utilidades ---
def to_device(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else x

def denormalize_tensor(tensor, mean=MEAN, std=STD):
    """tensor (C,H,W) o (B,C,H,W) normalizado -> pixel-space [0,1]"""
    m = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1) if tensor.dim()==4 else torch.tensor(mean, device=tensor.device).view(-1,1,1)
    s = torch.tensor(std,  device=tensor.device).view(1, -1, 1, 1) if tensor.dim()==4 else torch.tensor(std,  device=tensor.device).view(-1,1,1)
    return (tensor * s + m).clamp(0.0, 1.0)

def normalize_for_model(tensor, mean=MEAN, std=STD):
    m = torch.tensor(mean, device=tensor.device).view(-1,1,1)
    s = torch.tensor(std,  device=tensor.device).view(-1,1,1)
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

# --- main: ejemplo de uso y visualización ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cargar modelo y dataset (transform incluye Normalize)
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)
    preprocess = weights.transforms()  # incluye ToTensor + Normalize
    vis_pre = preprocess.transforms[:-1] if hasattr(preprocess, "transforms") else None
    # Si weights.transforms() no es compostable, puedes definir manualmente:
    # pre_norm = transforms.Compose([Resize(256), CenterCrop(224), ToTensor()])

    val_ds = ImageNet100ValDataset(VAL_DIR, transform=preprocess)
    img_tensor, label_idx = val_ds[0]            # img_tensor: normalizado (C,H,W)

    # 1) obtener gradiente en pixel-space (para sumar a la imagen en [0,1])
    grad_pixel = image_gradient(model, img_tensor, label_idx,
                                to_pixel_space=True, mean=MEAN, std=STD)  # (C,H,W)

    # 2) visualizaciones
    img_pixel = denormalize_tensor(img_tensor.unsqueeze(0)).squeeze(0)  # (C,H,W) en [0,1]
    grad_np = grad_pixel.permute(1,2,0).numpy()                         # (H,W,C)

    # Mostrar canales
    plt.figure(figsize=(12,4))
    for i, ch in enumerate(['R','G','B']):
        plt.subplot(1,3,i+1)
        chan = grad_np[:,:,i]
        norm = (chan - chan.min()) / (chan.max() - chan.min() + 1e-8)
        plt.imshow(norm, cmap='gray')
        plt.title(f'Grad {ch}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Mostrar mapa combinado (L2)
    grad_comb = torch.norm(grad_pixel, dim=0).numpy()
    grad_comb_norm = (grad_comb - grad_comb.min()) / (grad_comb.max() - grad_comb.min() + 1e-8)
    plt.figure(figsize=(5,5))
    plt.imshow(grad_comb_norm, cmap='viridis')
    plt.title('Grad L2 combinada')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    # 3) crear ejemplo de perturbación tipo FGSM (sumando signo en pixel-space)
    eps = 8/255
    img_perturbed = (img_pixel + eps * grad_pixel.sign()).clamp(0.,1.)

    # mostrar original vs perturbada
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].imshow(img_pixel.permute(1,2,0).numpy()); axs[0].set_title('Original'); axs[0].axis('off')
    axs[1].imshow(img_perturbed.permute(1,2,0).numpy()); axs[1].set_title('Original + FGSM'); axs[1].axis('off')
    plt.show()
    
    import utils as utils

    imagenet_classes = weights.meta["categories"]
    selected_indices_in_model = [imagenet_classes.index(labels[wnid].split(',')[0])
                                for wnid in selected_classes]
    model.eval()
    print(f"Perturbation with eps={eps}:")
    pred_wnid, nombre_legible, prob = utils.evaluar_imagen(model, img_tensor, selected_indices_in_model)
    print(f"Prediction on original image:")
    print(pred_wnid, nombre_legible, prob)
    pred_wnid, nombre_legible, prob = utils.evaluar_imagen(model, img_perturbed, selected_indices_in_model)
    print(f"Prediction on perturbed image:")
    print(pred_wnid, nombre_legible, prob)