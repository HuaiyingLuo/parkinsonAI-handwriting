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
>>> python -m pip install -r requirements.extra.txt
>>> python -m pip install -Ue .
```

To update the virtual environment with updated dependencies,run:
```
>>> python -m pip install -r requirements.txt
>>> python -m pip install -r requirements.extra.txt
```

## /data
dataset source: https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/
Create a folder named /data under the project root directory. Do not upload your dataset to GitHub. 
Add /data to your .gitignore file to prevent accidental commits.

## /EDA 

This is the workspace for Exploratory Data Analysis. All scripts are written in Jupyter Notebooks (.ipynb files).
Use this space to analyze data, visualize patterns, and document findings.