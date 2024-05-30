import os
import cv2
import random
from utils import resize_to_divisible, get_patches
import numpy as np

# Get all image paths
image_paths = [os.path.join('self_supervised_cropped', f) for f in os.listdir('self_supervised_cropped') if f.endswith('.jpg')]

for image_path in image_paths:
    # Load image
    image = cv2.imread(image_path)

    # Resize image to be divisible by 100
    image = resize_to_divisible(image, (100, 100))

    # Get patches
    patches = get_patches(image, (100, 100))
    patches = [patch for patch in patches if np.mean(patch) < 200]
    random.shuffle(patches)

    # Randomly rotate half of the patches
    num_patches = len(patches)
    indices_to_rotate = random.sample(range(num_patches), num_patches // 2)
    for idx in indices_to_rotate:
        patches[idx] = cv2.rotate(patches[idx], cv2.ROTATE_90_CLOCKWISE)

    # Save paired patches
    image_name = os.path.basename(image_path).split('.')[0]
    with open('Dataset_with_label/paired_patches.txt', 'a') as f:
        for i in range(0, len(patches) - len(patches) % 2 -2, 2):
            f.write(f'{image_name}_{i}.jpg {i in indices_to_rotate} {i+1 in indices_to_rotate}\n')
            cv2.imwrite(f'Dataset_with_label/first_half/{image_name}_{i}.jpg', patches[i])
            cv2.imwrite(f'Dataset_with_label/second_half/{image_name}_{i}.jpg', patches[i+1])
    