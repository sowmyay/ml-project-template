import os
from pathlib import Path


def main():
    project = input("What would you like to name your project? \n")
    if project == "":
        raise ValueError("Project name must not be empty")
    current_directory = Path(os.path.split(__file__)[0])
    script_path = os.path.join(current_directory.parent, "scripts", "pytorch-template.sh")
    template_path = os.path.join(current_directory.parent, "pytorch_template")
    os.system(f'sh {script_path} {project} {template_path}')


def parse(arg):
    if arg == "":
        return '\"\"'
    return arg


if __name__ == "__main__":
    main()
