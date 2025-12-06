import torch
from ImageNet100ValDataset import *
from torchvision.models import resnet34, ResNet34_Weights
from ataqueFGSM import *

def reducebit(x, bits=4):
    x = x.clone()

    x = x.clamp(0, 1)

    levels = 2 ** bits
    x = torch.round(x * (levels - 1)) / (levels - 1)

    return x.clamp(0, 1)

if __name__ == "__main__":
    dataset = ImageNet100ValDataset(VAL_DIR, transform=transform)

    transform_before_model = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.eval()

    # Contadores de accuracy
    acc_pure = 0
    acc_attack = 0
    acc_def7 = 0
    acc_def6 = 0
    acc_def5 = 0

    N = 5000   

    for i in range(N):
        # Cargar imagen y etiqueta
        x, y = dataset[i]
        x = x.unsqueeze(0) 
        y = torch.tensor([y])

        pred_pure = model(x)[:, selected_indices_in_model].argmax(dim=1)
        acc_pure += (pred_pure == y).item()

        #  Ataque FGSM 
        adv = generar_imagen_fgsm(model, x.squeeze(0), y.item(), 0.05)
        if adv.ndim == 3:
            adv = adv.unsqueeze(0)

        pred_attack = model(adv)[:, selected_indices_in_model].argmax(dim=1)
        acc_attack += (pred_attack == y).item()

        #  Desnormalizar 
        adv_denorm = denormalize_tensor(adv.squeeze(0))
        adv_denorm = adv_denorm

        #  DEFENSAS 

        # Bits 7
        x7 = reducebit(adv_denorm, bits=7)
        pred_def7 = model(x7)[:, selected_indices_in_model].argmax(dim=1)
        acc_def7 += (pred_def7 == y).item()

        # Bits 6
        x6 = reducebit(adv_denorm, bits=6)
        pred_def6 = model(x6)[:, selected_indices_in_model].argmax(dim=1)
        acc_def6 += (pred_def6 == y).item()

        # Bits 5
        x5 = reducebit(adv_denorm, bits=5)
        pred_def5 = model(x5)[:, selected_indices_in_model].argmax(dim=1)
        acc_def5 += (pred_def5 == y).item()

        print(f"[{i}/{N}] pure={pred_pure.item()} atk={pred_attack.item()} bit7={pred_def7.item()} bit6={pred_def6.item()} bit5={pred_def5.item()}")


    print("\nRESULTADOS:")
    print(f"Accuracy Puro:          {acc_pure / N:.4f}")
    print(f"Accuracy Ataque:        {acc_attack / N:.4f}")
    print(f"Accuracy Defensa bit 7: {acc_def7 / N:.4f}")
    print(f"Accuracy Defensa bit 6: {acc_def6 / N:.4f}")
    print(f"Accuracy Defensa bit 5: {acc_def5 / N:.4f}")
