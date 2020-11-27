import os
import argparse


def main():
    parser = argparse.ArgumentParser(prog="test")
    parser.add_argument("-n", "--name", type=str, required=True)
    args = parser.parse_args()
    os.system(f'sh pytorch-template.sh {args.name}')


if __name__ == "__main__":
    main()
