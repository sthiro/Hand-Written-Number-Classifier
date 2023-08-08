import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
import model_builder 
import image_transformer

device = "cuda" if torch.cuda.is_available() else "cpu"

## Make Model
model_0 = model_builder.CNN(input_shape=784, #One pixel per input neuron
                                 hidden_units=20,
                                 output_shape=10).to(device)

custom_image_path = "sample_images/number_0.jpeg"
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

final_image = image_transformer.transform_image(custom_image) # Transform an image as NN likes it !

## Evalute
model_path = "trained_model/CNN_Model_(1).pth"
model = model_0.to(device)  # Replace with your model class
model.load_state_dict(torch.load(model_path)) #Load pre-trained weights

model.eval()
with torch.inference_mode():
    
    target_image_pred = model(final_image.to(device)) # logits
    softMax_pred = torch.softmax(target_image_pred, dim=1) #Soft Max Layer
    pred = softMax_pred.argmax(dim=1) #Select the high value among the list


#Plot custom image
plt.imshow(final_image.permute(1, 2, 0), cmap = "gray") # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
plt.title(f"Predicting Number : {int(pred)} Probability : {softMax_pred.max()*100}")
plt.show()




