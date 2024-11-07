from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
from torchvision import transforms
from resnet50.model import Resnet50

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # directory for storing uploaded files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = Resnet50(num_classes=2).to(device)
    model.load_state_dict(torch.load('resnet50/models/resnet50_spiral_1103.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

model = load_model()

def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = transform(image).unsqueeze(0) # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件被上传
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            image = preprocess_image(filepath)
            with torch.no_grad():
                outputs = model(image)  
                predicted = torch.max(outputs, dim=1)[1]
            prediction = predicted.item()
            
            return render_template('result.html', prediction=prediction)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
