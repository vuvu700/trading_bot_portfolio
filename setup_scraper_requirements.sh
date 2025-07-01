#!/bin/bash

# Exit on error
set -e

# Check that a Python command is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <python_command> (e.g. python3.10)"
  exit 1
fi
pycommand=$1

echo "--> creating the venv"
$pycommand -m venv .venv
source .venv/bin/activate

# setup locals
echo "--> ensure locale fr_FR.UTF-8"
if ! locale -a | grep -q "fr_FR.utf8"; then
    sudo apt install locales
    sudo locale-gen fr_FR.UTF-8
    sudo update-locale
fi

# install packages
echo "--> install required packages"
#pip install --upgrade pip
pip install -r requirements_scrapper.txt
