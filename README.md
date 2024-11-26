# parkinsonAI-handwriting

## Setups
The virtual environment lets you install packages that are only used for your assignments and do not impact the rest of the system. If you choose venv, run the following command:

```
>>> python -m venv .venv
>>> source .venv/bin/activate
```

There are several packages used in this project, and you can install them in your virtual enviroment by running:

```
>>> python -m pip install -r requirements.txt
```

## Flask Web Application
### Run locally
Please check 'resnet50/models/resnet50_fusion_1111.pth' is fully downloaded. This file is managed by Git LFS (Large File Storage). To download this file, you need to have Git LFS installed and properly set up in your environment. 

Once you have LFS installed in your system, navigate to the repository directory and run the following command to fetch the actual content of the LFS files:

```
git lfs pull
```

Then run the command:
```
python app.py
```