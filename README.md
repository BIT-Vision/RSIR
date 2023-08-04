# RSIR
**R**ecurrent **S**pike-based **I**mage **R**estoration under General Illumination (ACM MM 2023)

RSIR is a fully supervised learning algorithm based on spike camera noise modeling and data synthesis.
It is used for reconstructing clean images from high-density spike stream captured under different lighting conditions.

# Overview
This is an official implementation of RSIR with Pytorch.

# Requirements
python=3.7

pytorch=1.13.1

cuda=11.1

opencv-python=4.7.0.68

tqdm

matplotlib

# Dataset
The simulated spike dataset:

The real-world spike dataset:
# Acknowledgement
This implementations are inspired by following projects:

- [EMVD] (https://github.com/Baymax-chen/EMVD)
