# install_dependencies.py

import subprocess

# List of dependencies to install
dependencies = [
    'pandas',
    'numpy',
]

# Install each dependency using pip
for package in dependencies:
    subprocess.call(['pip', 'install', package])

print("Dependencies installed successfully.")
