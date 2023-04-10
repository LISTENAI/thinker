from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# import struct
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Load image
image = Image.open("./demo/resnet28/apple.jpg")

# Define the preprocessing operations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
])

# Apply preprocessing operations
preprocessed_image = transform(image)

# Convert the preprocessed image to a NumPy array
preprocessed_image_np = preprocessed_image.numpy()

# Convert the preprocessed image data to int8
preprocessed_image_int8 = np.floor(preprocessed_image_np * 64 + 0.5).astype(np.int8)
preprocessed_image_int8.tofile("./demo/resnet28/apple_after_resize.bin")

