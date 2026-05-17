# Driver Distraction Detection using Personalized Machine Learning

## Associated Publication
Driver Distraction Detection using Personalized Machine Learning     
Achala Aponso, Craig Speelman, Michael N. Johnstone
Edith Cowan University, Joondalup, WA, Australia
Submitted to IEEE Access, 2026.

## Overview
This repository contains analysis code for comparing group-level and personalized machine learning models for cognitive driver distraction detection using multimodal physiological signals (EEG, GSR, HR).

## Models Implemented
### Group-Level (LOPO)
- slp_group.py
- mlp_group.py
- naive_bayes_group.py
- lstm_group.py
- gru_group.py
- transformer_group.py

### Personalized (Within-Subject)
- slp_individual.py
- mlp_individual.py
- naive_bayes_individual.py
- lstm_individual.py
- gru_individual.py
- transformer_individual.py

## Requirements
Python 3.12.7 (Anaconda distribution)
pip install -r requirements.txt

## Dataset
Dataset is publicly available at:
https://doi.org/10.5281/zenodo.20233645

## License
CC BY-NC 4.0 
