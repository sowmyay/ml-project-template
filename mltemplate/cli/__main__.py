import os
from pathlib import Path

def main():
    project = input("What would you like to name your project? \n")
    if project == "":
        raise ValueError("Project name must not be empty")
    name = parse(input("Enter author's first name \n"))
    surname = parse(input("Enter author's family name. Leave blank if not relevant \n"))
    email = parse(input("Enter email address \n"))

    current_directory = Path(os.path.split(__file__)[0])
    script_path = os.path.join(current_directory.parent, "scripts", "pytorch-template.sh")
    os.system(f'sh {script_path} {project} {name} {surname} {email}')

def parse(arg):
    if arg == "":
        return '\"\"'
    return arg

if __name__ == "__main__":
    main()
