# ml-project-template

This project is aimed to automate the boiler plate code for most machine learning projects.

It includes an executable shell script [pytorch-template.sh](pytorch-template.sh)

*TODO*:
Add boiler plate support for other languages
- [ ] tensorflow
- [ ] keras

# Usage
1. Download the executable script to your projects folder
2. Navigate to your projects folder containing the script
3. Run the script as follows. Replace `test` with project name of your choice -
```commandline
bash pytorch-template.sh test
```

This creates a project directory in your current folder similar to -
```markdown
template
├── Dockerfile.cpu
├── Dockerfile.gpu
├── Makefile
├── pyproject.toml
├── poetry.lock
├── notebooks
├── README.md
└── template
    ├── cli
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── predict.py
    │   └── train.py
    ├── __init__.py
    ├── models.py
    ├── datasets.py
    └── transforms.py
```
Checkout the `README.md` in the directory to see how to build and run the docker image.
You can also eventually add the scripts to train and predict your model here.
