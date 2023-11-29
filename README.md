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

# Additional download
You can download the dataset and model checkpoint from [Google Drive](https://drive.google.com/drive/folders/1oYGCuHLqJ8hH6kpQuyH0uDddhz1NefKz?usp=drive_link)

# Usage
## test
For test our model directly, please download checkpoint and data form the google drive first.
Then run `test.py` after making sure all from the `config.py` are correct. 

## train
If you want to retrain our model to adjust your own camera, please calibrate three uniform light first, and follow the details of our paper.
Retrain NIM module for the best performace or you can also use `nim.pt` with new $D_{dark}$, $Q_r$ and $R$ calculated by `cal_para` function.
The run `train.py` after making sure all from the `config.py` are correct.

# Acknowledgement
This implementations are inspired by following projects:

- [EMVD] (https://github.com/Baymax-chen/EMVD)
- [Swin-Transformer] (https://github.com/microsoft/Swin-Transformer)
