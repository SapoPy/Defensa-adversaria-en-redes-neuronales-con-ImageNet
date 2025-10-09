from ImageNet100ValDataset import *
from Modelos.ResNet50 import evaluate_resnet50
from Modelos.ResNet18 import evaluate_resnet18

if __name__ == "__main__":
    print(f"Precisión ResNet50 en validación: {evaluate_resnet50():.4f}")
    print(f"Precisión ResNet18 en validación: {evaluate_resnet18():.4f}")