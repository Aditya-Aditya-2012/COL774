#!/bin/bash

# Define the environment name and Python version
ENV_NAME="env_name3"
PYTHON_VERSION="3.10.9"

# Create a new conda environment with the specified Python version
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Install packages from requirements.txt using pip in the new environment
while read -r package; do
    conda run -n $ENV_NAME pip install "$package" || echo "Failed to install $package"
done < requirements.txt

# Provide feedback to the user
echo "Environment $ENV_NAME created with Python $PYTHON_VERSION and packages from requirements.txt installed."
