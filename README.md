# parkinsonAI-handwriting

## Overview
Parkinson’s Disease (PD) early identification is challenging due to subtle symptoms that mimic normal aging in its initial stages. This project aims to automatically identify early-stage PD patients by employing Convolutional Neural Networks (CNNs) to analyze handwriting dynamics, specifically spiral and meander movement exams.
Our goal is to assist doctors in resource-limited settings to diagnose PD rather than being a substitute for doctors, minimizing potential medical risks

### Dataset
In this project, we use both the [HandPD](https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/) and NewHandPD dataset, containing handwritten exams collected at Botucatu Medical School, São Paulo State University, Brazil in 2014. 
In total we have hand written images of 53 healthy individual(33.5%) and 105 patient individuals(66.5%).
The main task consists, essentially, in filling out a form composed by 4 spirals and 4 meanders, which are then cropped out from the form and stored in "jpg" image format.
We choose both the Meander and the Spiral dataset to build our classification model due to the sufficiency of data points in each dataset. 

### CNN Model
A ResNet50 model is employed as the core feature extractor for this classification task. 
The model is fine-tuned on the dataset by loading pre-trained weights and modifying the final classifier layer to suit the binary classification requirement. 

### Web Application
We developed a Streamlit app to provide a user-friendly interface for the model. Users can upload their handwriting images and get the classification results in just a few clicks.
Check out the web application at: [HandSight](https://handsight.streamlit.app/)


## Setups
The virtual environment lets you install packages that are only used for this project.

```
python -m venv .venv
source .venv/bin/activate
```

There are several packages used in the streamlit app, and you can install them in your virtual enviroment by running:

```
python -m pip install -r requirements.txt
```

## Flask Application
### Run locally
Please check 'resnet50/models/resnet50_fusion_1111.pth' is fully downloaded. This file is managed by Git LFS (Large File Storage). To download this file, you need to have Git LFS installed and properly set up in your environment. 

Once you have LFS installed in your system, navigate to the repository directory and run the following command to fetch the actual content of the LFS files:

```
git lfs pull
```

First install Flask:
```
pip install flask
```

Then run the command:
```
python app.py
```

## Streamlit Application
### Run locally
Please check 'resnet50/models/resnet50_fusion_1111.pth' is fully downloaded. This file is managed by Git LFS (Large File Storage). To download this file, you need to have Git LFS installed and properly set up in your environment. 

Once you have LFS installed in your system, navigate to the repository directory and run the following command to fetch the actual content of the LFS files:

```
git lfs pull
```

First install Streamlit:
```
pip install streamlit
```

Run the Streamlit app:
```
streamlit run HandSight.py
```

### Run on Streamlit Community Cloud
Try out the app at:
https://handsight.streamlit.app/

## Using docker to containerize the flask service 

Create a Dockerfile in the root directory of the project to build the docker image:

```
docker build -t handsight-app .
```

Run the docker container:
```
docker run -p 80:80 handsight-app
```

