FROM python:3.11.10-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
# Create the resnet50 folder and copy specific files into it
RUN mkdir -p /deploy/resnet50/
COPY ./resnet50/__init__.py /deploy/resnet50/
COPY ./resnet50/model_bagging.py /deploy/resnet50/
COPY ./resnet50/models/resnet50_fusion_1111.pth /deploy/resnet50/models/
# Copy the entire templates directory
COPY ./templates /deploy/templates/
# Copy the entire static directory
COPY ./static /deploy/static/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]

