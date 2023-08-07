# RSIR
**R**ecurrent **S**pike-based **I**mage **R**estoration under General Illumination (ACM MM 2023)

RSIR is a fully supervised learning algorithm based on noise modeling and data synthesis for spike camera.
It is used for reconstructing clean images from high-density spike stream captured under different lighting conditions.

This is an official implementation of RSIR with Pytorch.

# Requirements
python=3.7

pytorch=1.13.1

cuda=11.1

opencv-python=4.7.0.68

timm (for Swin Transformer)

tqdm

matplotlib

tensorboardX

# Dataset
The simulated spike dataset:

The real-world spike dataset:

# Checkpoint
The pretrained NIM module:[https://drive.google.com/file/d/17wP3isQpRTLQw4DmxlXl08aQOmr6wifF/view?usp=drive_link]

The best model:[https://drive.google.com/file/d/1i05zUfglBZFhrj3iXS0oWn0UDIFzjrZr/view?usp=drive_link]

# Acknowledgement
This implementations are inspired by following projects:

- [EMVD] (https://github.com/Baymax-chen/EMVD)
- [Swin-Transformer] (https://github.com/microsoft/Swin-Transformer)
