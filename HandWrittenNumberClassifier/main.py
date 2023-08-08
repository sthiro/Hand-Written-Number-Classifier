import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
import model_builder 
import image_transformer
import read_aloud
import cv2


## Most Repeated Prediction Function

def most_frequent(List):
    return max(set(List), key = List.count)

device = "cuda" if torch.cuda.is_available() else "cpu"

## Make Model
model_0 = model_builder.CNN()

## Evalute
model_path = "trained_model\CNN_Model_(1).pth"  #   uncomment for mac"HandWrittenNumberClassifier/trained_model/Trained_Cnn_model(2).pth"
model = model_0.to(device)  # Replace with your model class
model.load_state_dict(torch.load(model_path, map_location=device)) #Load pre-trained weights

## Open CV
cap = cv2.VideoCapture(0) # setup Webcam
old_predic = None
font = cv2.FONT_HERSHEY_SIMPLEX

prediction_list = []
text_cv = ""

while True:
    
    ret, frame = cap.read() # Read it
    
    frame_tensor = torch.from_numpy(frame).permute(2,0,1) # Make it Tensor are Rearrange
    final_image = image_transformer.transform_image_Evaluate(frame_tensor) # Transform an image as NN likes it !

    model.eval()
    with torch.inference_mode():
        
        target_image_pred = model(final_image.to(device)) # logits
        softMax_pred = torch.softmax(target_image_pred, dim=1) #Soft Max Layer
        pred = softMax_pred.argmax(dim=1) #Select the high value among the list

    cv2.putText(frame, 
            f'Number Prediction : {int(pred)} ', 
            (50, 50),  
            font,1,
            (0, 255, 255), 
            2, 
            cv2.LINE_4)
    

    cv2.imshow("Hand Written Number Prediction", frame) # Display it !

    prediction_list.append(int(pred))
    #print(len(prediction_list))

    if len(prediction_list) >= 75 and old_predic != most_frequent(prediction_list):

        old_predic = most_frequent(prediction_list) # Read Aloud
        #read_aloud.speak_sentence(f"Number {old_predic}") #Only works for mac

        prediction_list.clear()
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


##Uncomment to plot in Matplotlib

plt.imshow(final_image[0].permute(1, 2, 0), cmap = "gray") # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
plt.title(f"Prediction number : {int(pred)}")
plt.show()




