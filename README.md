# ml-project-template

This project is aimed to automate the boiler plate code for most machine learning projects.

It includes an executable shell script [ml-template-mkdir.sh](ml-template-mkdir.sh)

# Usage
1. Download or copy the test in the executable script to your projects folder
2. Navigate to your projects folder containing  `ml-template-mkdir.sh`
3. Execute the script as follows. Replace test with the desired name for your repository -
```commandline
bash ml-template-mkdir.sh test
```

This creates a project directory in your current folder similar to -
```markdown
test
├── bin
│   └── test
├── Dockerfile.cpu
├── Dockerfile.gpu
├── Makefile
├── notebooks
├── README.md
├── requirements.in
└── test
    ├── cli
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── predict.py
    │   └── train.py
    ├── datasets.py
    ├── __init__.py
    ├── models.py
    └── transforms.py
```
Checkout the `README.md` in the directory to see how to build and run the docker image.
You can also eventually add the scripts to train and predict your model here.
