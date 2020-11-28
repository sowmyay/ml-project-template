# ML Template

ML template is an easy to use tool to automate the boiler plate code for most machine learning projects.

This tool creates a user-oriented project architecture for machine learning projects.

Modify the code under `#TODO` comments in the template project repository to easily adapt the template to your use-case.

# How to use it?
1. Install the package as - `pip install mltemplate`
2. Then, simply run `mltempate` from your terminal and follow the prompts

And Voila! 

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
All you have to do next is -
1. Head to `template/datasets.py` and modify create a new dataset that will work for your use case
2. Navigate to `template/models.py` and create a new model class with your sota (or not) architecture
3. In `template/transforms.py` add transforms such as Normalizer, Denormalize etc.
4. Follow the `TODO` steps in `template/cli/train.py` and `template/cli/predict.py` to make the necessary changes

Checkout the `README.md` in the `template` directory for further instructions on how to train, predict and also monitor your loss plots using tensor board.

# Future Work
Currently this package only supports boilerplate creation for ML projects in `pytorch`

We plan to support `tensorflow` in the future.

## License
Copyright © 2020 Sowmya Yellapragada

Distributed under the MIT License (MIT).