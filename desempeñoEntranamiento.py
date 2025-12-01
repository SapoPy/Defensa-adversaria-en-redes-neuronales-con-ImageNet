from utils import *
import torch
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn
from ataqueFGSM2 import *
from ataqueRFGSM2 import *
from ataquePDG2 import *
from AtaquesGaussianos2 import *

if __name__ == "__main__":

    # 1. Crear arquitectura EXACTA usada al entrenar
    model = resnet34(weights=None)  # NO cargar pesos ImageNet

    # 2. Cargar pesos entrenados
    state_dict = torch.load("resnet34_retrained_fl (1).pth", map_location="cpu")
    model.load_state_dict(state_dict)

    # 3. Mover a GPU si existe
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 4. Modo evaluación
    model.eval()

    VAL_DIR = "3 clases"

    epsilons = 0.05
    alpha = 2/255
    k = 2
    sigma = 2/255

    # 5. Evaluación o ataque
    acc_top1, acc_top5 = global_evaluate(
        model,
        None,                # si FGSM es interno deja None
        [epsilons, alpha],
        VAL_DIR
    )
    print(f"\nRe entrenado")
    print(f"Acc top 1 original: {acc_top1}")
    print(f"Acc top 5 original: {acc_top5}")

        # 5. Evaluación o ataque
    acc_top1, acc_top5 = global_evaluate(
        model,
        generar_imagen_PGD,                # si FGSM es interno deja None
        [epsilons, alpha, k],
        VAL_DIR
    )
    print(f"\n PGD")
    print(f"Acc top 1: {acc_top1}")
    print(f"Acc top 5: {acc_top5}")

        # 5. Evaluación o ataque
    acc_top1, acc_top5 = global_evaluate(
        model,
        generar_imagen_fgsm,                # si FGSM es interno deja None
        epsilons,
        VAL_DIR
    )
    print(f"\n FGSM")
    print(f"Acc top 1: {acc_top1}")
    print(f"Acc top 5: {acc_top5}")

        # 5. Evaluación o ataque
    acc_top1, acc_top5 = global_evaluate(
        model,
        generar_imagen_rfgsm,                # si FGSM es interno deja None
        [epsilons, alpha],
        VAL_DIR
    )
    print(f"\n RFGSM")
    print(f"Acc top 1: {acc_top1}")
    print(f"Acc top 5: {acc_top5}")