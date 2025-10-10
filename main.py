from ImageNet100ValDataset import *
from utils import *
from AtaquesGaussianos import *

if __name__ == "__main__":
    apply_noise_to_class("n01440764", "val_noisy_015/n01440764", std = 0.15)
    apply_noise_to_class("n01847000", "val_noisy_015/n01847000", std = 0.15)
    apply_noise_to_class("n01883070", "val_noisy_015/n01883070", std = 0.15)