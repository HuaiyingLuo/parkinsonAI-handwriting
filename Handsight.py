import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from resnet50.model_bagging import FusionModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = FusionModel(num_classes=2).to(device)
    model.load_state_dict(torch.load('resnet50/models/resnet50_fusion_1111.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

model = load_model()

def preprocess_image(image):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def main():
    st.title("Welcome to HandSight")
    st.write("HandSight is designed to assist in the early diagnosis of Parkinson's Disease (PD) by analyzing hand-drawn images using AI.")

    st.write("Draw a spiral and a meander from the inside out. Examples are as follows:")
    col1, col2 = st.columns(2)
    with col1:
        st.image("static/spiral_example.jpg", caption="Spiral Example", use_container_width=True)
    with col2:
        st.image("static/meander_example.jpg", caption="Meander Example", use_container_width=True)
    
    file1 = st.file_uploader("Upload the SPIRAL image", type=["jpg", "jpeg", "png"])
    file2 = st.file_uploader("Upload the MEANDER image", type=["jpg", "jpeg", "png"])
    
    if file1 and file2:
        image1 = Image.open(file1)
        image2 = Image.open(file2)
        
        meander = preprocess_image(image1)
        spiral = preprocess_image(image2)
        
        with torch.no_grad():
            outputs = model(meander, spiral)
            predicted = torch.max(outputs, dim=1)[1]
        
        prediction = predicted.item()
        st.header("Classification Result")
        if prediction == 0:
            st.write("Parkinson's Disease: :green[NEGATIVE]")             
        else:
            st.write("Parkinson's Disease: :red[POSITIVE]") 
               
        st.write("Please note that this result is only an assistive tool for early diagnosis. It is important to consult a neurologist for a comprehensive and accurate diagnosis.")

if __name__ == '__main__':
    main()
