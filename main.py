from ImageNet100ValDataset import *
from utils import *
from torchvision.models import resnet34, ResNet34_Weights
if __name__ == "__main__":
    val_dataset = ImageNet100ValDataset("FGSM_out", transform=None)
    weights = ResNet34_Weights.DEFAULT