# A network score-based metric to optimize the quality assurance of automatic target segmentation 

This repository contains the code for the paper "A network score-based metric to optimize the quality assurance of automatic target segmentation", currently under review.

# Reproducing results of the article
You can reproduce the results of the article with the script his_main.py
* To get the score-based metrics, run cell 1.
* To get the correlation results from Tables 3 and S3, run cell 2.
* To plot figure 2, run cell 3.
* To get the AUC curves, run cell 4.
* To get scatter plots, run cell 5.

# Information about trained models
The nnU-Net was used as segmentation framework for this project (https://github.com/MIC-DKFZ/nnUNet). The trained models and JSON files related to its training are stored in the folder segmentation_models.
