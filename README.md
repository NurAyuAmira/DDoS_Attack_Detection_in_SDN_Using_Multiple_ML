# DDoS_Attack_Detection_in_SDN_Using_Multiple_ML

This repository contains the code and dataset used for the detection of DDoS attacks in Software Defined Networking (SDN) environments using various machine learning models. The project focuses on implementing and comparing the performance of Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Naive Bayes (NB), and an Ensemble model. The dataset used for this study is the CIC-DDoS2019 dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Feature Selection](#feature-selection)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Visualizations](#visualizations)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Running on Google Colab](#running-on-google-colab)
- [Contributing](#contributing)


## Project Overview
This project aims to enhance the detection of DDoS attacks in SDN environments by leveraging machine learning techniques. The study compares the performance of different models to determine the most effective approach for accurate and efficient DDoS detection.

## Dataset
The dataset used in this project is the CIC-DDoS2019 dataset. The dataset has been preprocessed and cleaned to remove any inconsistencies and ensure it is suitable for training the models.

## Models Implemented
- **Support Vector Machine (SVM)**: Evaluated with linear, polynomial, and RBF kernels.
- **K-Nearest Neighbors (KNN)**: Tested with Euclidean, Manhattan, and Cosine distance metrics.
- **Naive Bayes (NB)**: Gaussian Naive Bayes.
- **Ensemble Model**: Combines SVM, KNN, and NB using soft voting.

## Feature Selection
Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) were used for feature selection to improve model accuracy and computational efficiency. PCA was used for SVM, KNN, and the Ensemble model, while LDA was used for Naive Bayes. The feature selection process helps in reducing the dimensionality of the dataset while retaining the most important features.

## Hyperparameter Tuning
Halving Grid Search was used for hyperparameter tuning to find the best parameters for each model, optimizing their performance.

## Results
The results showed that the Ensemble model achieved the highest accuracy of 98.29%, outperforming individual models. The performance metrics for each model are as follows:
- **SVM (RBF kernel)**: 94.67% accuracy
- **KNN (Manhattan distance)**: 93.37% accuracy
- **Naive Bayes**: 80.19% accuracy
- **Ensemble Model**: 98.29% accuracy

## Visualizations
Several visualizations are provided to illustrate the performance of the models, the impact of feature selection, and the effect of different hyperparameter configurations.

## Getting Started
To get started, clone this repository to your local machine and install the required dependencies.


    git clone https://github.com/your-username/DDoS_Attack_Detection_in_SDN_Using_Multiple_ML.git
    cd DDoS_Attack_Detection_in_SDN_Using_Multiple_ML
    pip install -r requirements.txt

## Usage 
Run the Jupyter notebooks to explore the data, train the models, and evaluate their performance.
- Data Understanding and Preparation: 2640_Data_Understanding_And_Preparation.ipynb
- Model Training and Evaluation: 2640_SVM_KNN_NB_Ensemble.ipynb
- Result Visualization: result_visualization.ipynb
- Streamlit App (60:40 Split): 2640_Streamlit_60_40.ipynb
- Streamlit App (70:30 Split): 2640_Streamlit_70_30.ipynb

## Running on Google Colab

To run this project on Google Colab, follow these steps:

1.  Open Google Colab in your browser.
2.  Select "File" > "Open notebook".
3.  Go to the "GitHub" tab.
4.  Enter the GitHub repository URL: `https://github.com/NurAyuAmira/DDoS_Attack_Detection_in_SDN_Using_Multiple_ML/blob/main/README.md`.
5.  Select the notebook you want to run from the list.
6.  After the notebook opens, run the first cell to clone the repository into the Colab environment:

    `!git clone https://github.com/your-username/DDOS-Detection-SDN.git
    %cd DDOS-Detection-SDN` 
    
7.  Install the required dependencies:
    
    `!pip install -r requirements.txt` 
    
8.  Proceed with running the remaining cells in the notebook.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your changes.

