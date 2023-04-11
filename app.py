import gradio as gr
import os
import torch

from timeit import default_timer as timer
from typing import Tuple, Dict
from torchvision import transforms

from models.model import TinyVGG

# Setup class names
with open("models/labels.txt", "r") as f: # reading them in from class_names.txt
    class_names = [urdu_character.strip() for urdu_character in  f.readlines()]
    
### 2. Model and transforms preparation ###    
model_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    
    transforms.ToTensor(),
])

# Recreate an instance of TinyVGG
model_0 = TinyVGG(input_shape=1, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(class_names))

# Load saved weights
model_0.load_state_dict(
    torch.load(
        f='models/saved/01_pytorch_workflow_model_0.pth',
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img_transform = model_transform(img).unsqueeze(dim=0)
    
    # Put model into evaluation mode and turn on inference mode
    model_0.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(model_0(img_transform), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "Urdu Characters Vision ☪︎👁"
description = "An TinyVGG feature extractor computer vision model to classify images of urdu characters [23 different classes]"

# Create examples list from "static/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create Gradio interface 
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
)

# Launch the app!
demo.launch()