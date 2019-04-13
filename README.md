# Neural Style Transfer

This repository contains the code for my implementation of the Neural Style Transfer paper by Gatys et al.

The model in question is a pretrained VGG19 that is able to take a content image and render it in the style of given style image.

### Setup and Training

See the `requirements.txt` for dependecies related to Python. After cloning the repository, you can install all these requirements using the `make` command. It is recommended to create a **virtual environment** for this purpose.
Run the **main.py** script to start the training run with the sample images.
The model will run for 20000 iterations before saving the new image. From my experiments, this should take ~15 mins on a P100 GPU.

### References

- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
