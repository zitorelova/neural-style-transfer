# Neural Style Transfer

This repository contains the code for my implementation of the Neural Style Transfer paper.
The implementation is based on this [paper](https://arxiv.org/abs/1508.06576) by Gatys et al. 

### Setup 

After cloning this repository, place your content and style images inside the data folder.
Run the **main.py** script and set the content and style arguments as your content and style images.
The model will run for 20000 iterations before saving the new image. From my experiments, this should take ~15 mins on a P100 GPU.

