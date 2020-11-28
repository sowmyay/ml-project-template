import os

def main():
    project = input("What would you like to name your project? \n")
    if project == "":
        raise ValueError("Project name must not be empty")
    name = parse(input("Enter author's first name \n"))
    surname = parse(input("Enter author's family name. Leave blank if not relevant \n"))
    email = parse(input("Enter email address \n"))
    os.system(f'sh mltemplate/scripts/pytorch-template.sh {project} {name} {surname} {email}')

def parse(arg):
    if arg == "":
        return '\"\"'
    return arg

if __name__ == "__main__":
    main()
