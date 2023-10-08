#!/bin/bash

echo "Creating ML template for $1"
path_to_template="$2"
template_name="$3"

cp -r "$path_to_template" "$1"
mv "$1/$template_name" "$1/$1"

# sed definition differs between linux and macos systems
echo "$OSTYPE"
if [[ "$OSTYPE" == "linux"* || "$OSTYPE" == "darwin"* ]]; then
  find "$1" -iname "*.py" -type f -exec sed -i '' "s/$template_name/$1/gi" {} \;
  sed -i '' "s/$template_name/$1/g" "$1"/Makefile "$1"/README.md
else
  find "$1" -iname "*.py" -type f -exec sed -i "s/$template_name/$1/gi" {} \;
  sed -i "s/$template_name/$1/g" "$1"/Makefile "$1"/README.md
fi
echo "Voila!"
