# install_dependencies.py

import subprocess

# List of dependencies to install
packages = [
    'pandas',
    'matplotlib',
    'seaborn',
    'scikit-learn',
    'scipy'
]

# Install each dependency using pip
for package in packages:
    subprocess.call(['pip3', 'install', package])

print("Dependencies installed successfully.")


# # install_dependencies.py
# import subprocess

# # List of dependencies to install
# packages = [
#     'pandas',
#     'matplotlib',
#     'seaborn',
#     'scikit-learn',
#     'scipy'
# ]

# # Install each dependency using pip if it's not already installed
# for package in packages:
#     try:
#         # Check if the package is already installed
#         subprocess.check_call(['pip3', 'show', package])
#     except subprocess.CalledProcessError:
#         # If the package is not installed, install it
#         subprocess.call(['pip3', 'install', package])
#     else:
#         print(f"{package} is already installed.")

# print("Dependencies installed successfully.")
