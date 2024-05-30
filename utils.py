import cv2
import numpy as np

def resize_to_divisible(image, patch_window):
	# Resize the image to be divisible by the patch window
	h, w = image.shape[:2]
	h = h - h % patch_window[0]
	w = w - w % patch_window[1]
	return cv2.resize(image, (w, h))

def get_patches(image, patch_window):
	# Get patches from the image
	h, w = image.shape[:2]
	patch_h, patch_w = patch_window
	patches = []
	for i in range(0, h, patch_h):
		for j in range(0, w, patch_w):
			patch = image[i:i+patch_h, j:j+patch_w]
			patches.append(patch)
	return patches

def patch_to_image(patches, image_shape, patch_window):
	# Combine patches to form the image
	h, w = image_shape
	patch_h, patch_w = patch_window
	image = np.zeros((h, w), dtype=np.float32)
	idx = 0
	for i in range(0, h, patch_h):
		for j in range(0, w, patch_w):
			image[i:i+patch_h, j:j+patch_w] = patches[idx]
			idx += 1
	return image


def extract_windows(image, window_size):
	# Pad the image
	padded_image = np.pad(image, ((window_size[0]//2, window_size[0]//2), (window_size[1]//2, window_size[1]//2), (0, 0)), mode='constant')
	
	# Extract windows
	windows = []
	for i in range(window_size[0]//2, padded_image.shape[0]-window_size[0]//2):
		for j in range(window_size[1]//2, padded_image.shape[1]-window_size[1]//2):
			window = padded_image[i-window_size[0]//2:i+window_size[0]//2, j-window_size[1]//2:j+window_size[1]//2, :]
			windows.append(window)
		
	return windows
