import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor

Evalute_transform = transforms.Compose([
 
    transforms.Grayscale(), # Make it Gray scale
    transforms.Lambda(lambda x: 1.0 - x), # Make the image inverted
    transforms.Resize((28, 28)), # resize the size
    

])


train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.ToTensor(), # use ToTensor() last to get everything between 0 & 1

])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    ToTensor(),

])



def transform_image_Evaluate(frame): # Input frame is tensor
    
    frame = frame / 255
    custom_image_transformed = Evalute_transform(frame) #Transfrom using EvaluteTransformer
    custom_image_transformed = torch.where(custom_image_transformed > 0.6, 1.0, 0.0) # Make it binary
    #print(final)
    
    final = flood_fill_from_corners(custom_image_transformed[0])
    
    return custom_image_transformed.unsqueeze(dim=0) #custom_image_transformed

def train_image_transform (frame): # Transform image for training purpose
    return train_transforms(frame)

def test_image_transform (frame): # Transform image for testing purpose
    return test_image_transform(frame)


def flood_fill_from_corners(binary_image):
   
    binary_image_np = (binary_image.numpy() * 255).astype(np.uint8)
    binary_image_np = binary_image_np  # Reshape to a 2-dimensional array
    flood_filled_images = []
    height, width = 28, 28

    corners = [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]

    for corner in corners:
        binary_image_copy = binary_image_np.copy()  # Create a copy of the original binary image
        if binary_image_copy[corner[1], corner[0]] == 255:  # Check if the corner pixel is white (foreground)
            # Update the mask size and initialize the mask with all zeros
            flood_fill_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
            # Set the flood fill area to 1 (white) because cv2.floodFill requires a non-zero value for the flood fill area
            flood_fill_mask[1:-1, 1:-1] = 1
            # Perform the flood fill operation
            cv2.floodFill(binary_image_copy, flood_fill_mask, corner, 0)  # Fill with black (0)

        flood_filled_image = torch.tensor(binary_image_copy / 255.0)
        flood_filled_images.append(flood_filled_image)

    return torch.stack(flood_filled_images).unsqueeze(dim=0)
# Example usage:
# Assuming binary_image is a torch tensor representing a 2D binary image (0 or 1)
# flood_filled_result = flood_fill_from_corners(binary_image)



