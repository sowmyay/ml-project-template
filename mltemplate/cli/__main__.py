import os
import argparse
from pathlib import Path
from enum import Enum


class Language(Enum):
    pytorch = 1
    keras = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Language[s]
        except KeyError:
            raise ValueError()


def main():
    parser = argparse.ArgumentParser(prog="mltemplate")
    parser.add_argument("project", type=str, help="What would you like to name your project?")
    parser.add_argument('--language', default="pytorch", type=Language.from_string, choices=list(Language),
                        help="Choose the programming language for your project")
    args = parser.parse_args()
    current_directory = Path(os.path.split(__file__)[0])
    script_path = os.path.join(current_directory.parent, "scripts", "template.sh")
    template_name = f"{args.language}_template"
    template_path = os.path.join(current_directory.parent, template_name)
    os.system(f'sh {script_path} {args.project} {template_path} {template_name}')


if __name__ == "__main__":
    main()
