import torch
from ImageNet100ValDataset import *
from utils import * 

def generar_imagen_gaussian(model, imgs, labels, args):
    sigma = args[0]

    imgs, labels, single = ensure_batch(imgs, labels)

    noise = torch.randn_like(imgs) * sigma
    adv = imgs + noise
    adv = adv.clamp(-3, 3)

    return adv[0] if single else adv



def evaluate_gaussian_attack(model, val_dir, sigma):
    return global_evaluate(model, generar_imagen_gaussian, [sigma], val_dir)



if __name__ == "__main__":
    from torchvision.models import resnet34, ResNet34_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)

    VAL_DIR = "3 clases"
    val_ds = ImageNet100ValDataset(VAL_DIR, transform=transform)

    sigma = 0.05


    acc_top1, acc_top5 = global_evaluate(model, generar_imagen_gaussian, [sigma], VAL_DIR)
    print(f"Precisión top 1 (gaussian on-the-fly): {acc_top1:.4f}")
    print(f"Precisión top 5 (gaussian on-the-fly): {acc_top5:.4f}")
    
    #metodo=None
    #acc_top1, acc_top5 = global_evaluate(model, metodo, [sigma], VAL_DIR)
    #print(f"Precisión top 1 (raw): {acc_top1:.4f}")
    #print(f"Precisión top 5 (raw): {acc_top5:.4f}") 