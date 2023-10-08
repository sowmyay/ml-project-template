#!/bin/bash

echo "Creating ML template for $1"
cp -r pytorch_template "$1"
mv "$1"/pytorch_template "$1/$1"
find "$1" -iname "*.py" -type f -exec sed -i '' "s/pytorch_template/$1/gi" {} \;
sed -i '' "s/pytorch_template/$1/g" "$1"/Makefile "$1"/README.md