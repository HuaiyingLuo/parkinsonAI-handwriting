from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
from torchvision import transforms
from resnet50.model_bagging import FusionModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # directory for storing uploaded files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = FusionModel(num_classes=2).to(device)
    model.load_state_dict(torch.load('resnet50/models/resnet50_fusion_1111.pth', map_location=device, weights_only=True))
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
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({"error": "Both files are required"}), 400
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        if file1 and file2:
            filename1 = secure_filename(file1.filename)
            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file1.save(filepath1)

            filename2 = secure_filename(file2.filename)
            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            file2.save(filepath2)

            meander = preprocess_image(filepath1)
            spiral = preprocess_image(filepath2)
            with torch.no_grad():
                outputs = model(meander, spiral)  
                predicted = torch.max(outputs, dim=1)[1]
            prediction = predicted.item()
            
            return render_template('result.html', prediction=prediction)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
