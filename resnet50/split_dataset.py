import os
from shutil import copy
import random
 
 
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)
 
data_type = 'meander'
file = f'resnet50/handwritten_dataset/{data_type}'
classifier_class = [cla for cla in os.listdir(file) if os.path.isdir(os.path.join(file, cla))]
mkfile(f'resnet50/handwritten_dataset/{data_type}/train')
for cla in classifier_class:
    mkfile(f'resnet50/handwritten_dataset/{data_type}/train/'+cla)
 
mkfile(f'resnet50/handwritten_dataset/{data_type}/val')
for cla in classifier_class:
    mkfile(f'resnet50/handwritten_dataset/{data_type}/val/'+cla)
 
split_rate = 0.25
for cla in classifier_class:
    cla_path = file + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = f'resnet50/handwritten_dataset/{data_type}/val/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = f'resnet50/handwritten_dataset/{data_type}/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    print()
 
print("processing done!")
