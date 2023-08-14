# Capstone Project: Black Forest Forecasters
<p align = "center"> <img src="./images/Capstone_presentation_Title.jpeg" />

This repository is associated with a four weeks long project, the so called "**Capstone Project**", that crowns our data science bootcamp 2023 (May to August) at [*neufische GmbH*](https://www.neuefische.de/). 

## Project Overview

In this project, we tackle the problem of classifying individual trees using remote sensing data (LiDAR & True Orthophotos) as well as additional geophysical data (ground elevation, slope exposure, slope angle, vegetation height). We aim at the creation of an automatized model that is trained on the images of pre-labed individual trees and is then able to classify tree species.
As convolutional neural networks (CNNs) are especially suited to handle image data, we will focus on them to perform the classification of tree species. The data of this project is provided by the black forest national park and comprises 1700 labeled trees and their corresponding images. 

At best, we would like to create a data product that reads in a map of a forested area, segments individual trees and then classifies them using our convolutional neural network. This data product would then be applicable to any map with a similar vegetation and tree species using our trained model.

Our first step was to extract tree images from the Orthophotos and other dataproducts and store them and the associated tree labels in a `.npz` file. Afterwards we sorted the trees into 4 categories in accordance with our stakeholder request. The the data gets preprocessed by translating the labels into One-Hot encoded representations and the images are normalized, augmented and oversampled to account for the imbalance in the dataset. For the modeling we used as a baseline a DNN with an accuracy of XXX precent. For our more advanced models we used a CNN with the Ortho-images only and an ensemble model combining the Ortho-images with VEGHEIGHT,DENSITY,... data. 
Our best model achieves an accuracy of XXX precent.

## Setup
This project was created in Python `3.11.3`. To create an environment you can run the following code in your terminal. Or install the needed libraries and dependencies from the [requirements.txt-file](./requirements.txt).
```zsh
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
**Note:** The data that was used is not included in this repo.
