### Importing Packages

import dataloader 
import model_builder
import torch
from torch import nn
import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
# Import accuracy metric
from helper_functions import accuracy_fn # Note: could also use torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device using for training : {device}")

## Initilaize Mode

model_0 = model_builder.CNN().to(device)

## Optimizer and Loss Function
model_path = "CNN_Model2.pth"
model = model_0.to(device)  # Replace with your model class
model.load_state_dict(torch.load(model_path, map_location=device)) #Load pre-trained weights

loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)


### Train Function

train_loss_result = []
test_loss_result = []
train_acc_result = []
test_acc_result = []

#compiled_model = torch.compile(model_0) # Compile Model ** Pytorch V2 **   
compiled_model = model_0 # Its not using torch.compile()
train_dataloader = dataloader.train_dataloader
test_dataloader = dataloader.test_dataloader

def train(epochs,
          train_dataloader,
          test_dataloader):

  # Create training and testing loop
  for epoch in range(epochs):

      train_loss = 0
      # Add a loop to loop through training batches
      for batch, (X, y) in enumerate(train_dataloader):

          X = X.to(device)
          y = y.to(device)

          X = torch.where(X > 0.425, 1.0, 0.0) # make it binary

          compiled_model.train()
          # 1. Forward pass

          y_pred = compiled_model(X)

          # 2. Calculate loss (per batch)
          loss = loss_fn(y_pred, y)
          train_loss += loss # accumulatively add up the loss per epoch

          # 3. Optimizer zero grad
          optimizer.zero_grad()

          # 4. Loss backward
          loss.backward()

          # 5. Optimizer step
          optimizer.step()

          # Print out how many samples have been seen
          # if batch % 100000 == 0:
          #     print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

      # Divide total train loss by length of train dataloader (average loss per batch per epoch)
      train_loss /= len(train_dataloader)
      train_loss_result.append(float(train_loss))


      ### Testing
      # Setup variables for accumulatively adding up loss and accuracy
      test_loss, test_acc = 0, 0
      compiled_model.eval()

      with torch.inference_mode():

          for X, y in test_dataloader:

              X = X.to(device)
              y = y.to(device)

              X = torch.where(X > 0.425, 1.0, 0.0) # make it Binary

              # 1. Forward pass

              test_pred = compiled_model(X)

              # 2. Calculate loss (accumatively)

              test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

              # 3. Calculate accuracy (preds need to be same as y_true)
              test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

          # Calculations on test metrics need to happen inside torch.inference_mode()
          # Divide total test loss by length of test dataloader (per batch)
          test_loss /= len(test_dataloader)
          test_loss_result.append(float(test_loss))

          # Divide total accuracy by length of test dataloader (per batch)
          test_acc /= len(test_dataloader)
          test_acc_result.append(float(test_acc))

      print(f"Epoch : {epoch} Test_Loss : {test_loss} Train_loss : {train_loss}")
      model_path = f'trained_model/model_epoch_{epoch+1}.pth'
      torch.save(compiled_model.state_dict(), model_path)


  return train_loss_result, test_loss_result, test_acc_result


## Calling Training Function

train_loss_result, test_loss_result, test_acc_result = train(1000 ,train_dataloader, test_dataloader)
model_path = "trained_model/Final_Model"

torch.save(model_0.state_dict(), model_path)

print(train_loss_result,test_acc_result)


### Evaluating

## Ploting Loss Results
import matplotlib.pyplot as plt

plt.plot(train_loss_result)
plt.title('train_loss_result')
plt.plot(test_loss_result)
plt.title('test_loss_result')
plt.show()

## Plotting Accuracy Results

plt.plot(test_acc_result)
plt.title('test_acc_result')
plt.show()

print(f"Most Efficiant Model --> {test_loss_result.index(min(test_loss_result))}")