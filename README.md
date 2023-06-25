# Drug Discovery using Virtual Screening
This project aims to develop a computational approach for drug discovery using virtual screening techniques. Virtual screening is a computational method that involves screening large chemical libraries to identify potential drug candidates. It can significantly reduce the time and cost required for the initial stages of drug discovery.


https://github.com/Wa-lead/Virtual_screnning_using_CNNs/assets/81301826/f22e3a86-7025-4ca3-bb5b-7bba8055d0bd


# Project Overview
The project consists of several steps:

* Data Preparation: The project utilizes a dataset of chemical compounds and their corresponding biological activities. The dataset is preprocessed and featurized using the ChemCeption model.


![Alt text](images/Featurizer.png)

* Model Building: The ChemCeption model, which is based on the InceptionV3 architecture, is used to predict the biological activity of the chemical compounds. The model is trained using the preprocessed data.
* Model Evaluation: The trained model is evaluated using a separate test dataset. The performance of the model is measured using metrics such as ROC curve and AUC score.


![Alt text](images/ROC.png)


* Kernel Visualization: The project includes techniques for visualizing the model's decision-making process, such as kernel visualization and Grad-CAM.

![Alt text](images/kernel.png)


![Alt text](images/GradCAM.png)

# Hyperparameter Optimization 
We used the **Hpbandster** package for optimizing the model, which combines hyperband and bayesian optimization.

![Alt text](images/hpbandster.png)


# Installation
To run the project, follow these steps:

Clone the repository:
> git clone https://github.com/your_username/project.git

Install the required dependencies: 
> pip install -r requirements.txt

# Evaluation
The project achieves an ROC score of 0.69, indicating a good performance in predicting the biological activity of chemical compounds. 

# Usage
Run the inference.ipynb
![Capture-2023-05-17-132002](https://github.com/Wa-lead/Chemception_using_transfer_learning/assets/81301826/a4ad8699-cafb-49f8-96a7-dcf43304358a)
