# Threat Detection in Cyber Security Using AI

## Overview

The "Threat Detection in Cyber Security Using AI" project aims to develop a threat detection system using machine learning algorithms. The project consists of several steps, each of which contributes to the overall goal of enhancing cyber security. Here's an overview of each step:

### Step 1: Data Preprocessing (PreProcessing.ipynb)

- This step involves data preprocessing to prepare the dataset for machine learning.
- The dataset used is the CIC-IDS2017 dataset, which should be stored in the "CSVs" folder located in the same directory as the program.
- You can access the dataset files [here](https://www.unb.ca/cic/datasets/ids-2017.html).

### Step 2: Attack Data Filtering (AttackDivision.ipynb)

- In this step, the program uses the "all_data.csv" file to create attack-specific files.
- These attack files are then saved in the "./attacks/" directory for further analysis.
- The dataset contains a total of 12 attack types, and this step separates them for individual examination.

### Step 3: Feature Selection and Machine Learning (FeatureSelection.ipynb)

- This step focuses on feature selection for the attack files created in Step 2.
- The program identifies the four features with the highest weight for each file.
- These selected features are used as input for machine learning algorithms.

### Step 4: Machine Learning Algorithm Evaluation (MachineLearningSep.ipynb)

- The final step applies seven machine learning algorithms to each attack file multiple times for robust evaluation.
- Results of these operations are displayed on the screen and saved in the file "./attacks/results_1.csv".
- Additionally, box and whisker graphics representing the results are generated.

### Step 5: Handle class imbalance 

- It’s when one category in your dataset has way more samples than the others, making it harder for the model to learn fairly.
- Each machine learning algorithm is trained using the attack-specific datasets with selected high-weight features.
- Time taken for training and inference is also recorded to assess computational efficiency.

## Step 6: Model Comparision and Best Model Comparision 

- Models are compared across multiple attack types to assess generalization and robustness.
- Visualization through box and whisker plots provides a clear picture of how each model performs across various metrics.
- Seven ML algorithms are compared: Random Forest, Decision Tree, SVM, KNN, Logistic Regression, Naive Bayes, and Gradient Boosting.

## Step 7: Advanced Evaluation For Best Model

- After training, we use advanced evaluation metrics like precision, recall, F1-score, and confusion matrix to deeply analyze how well the model is performing.
- Model robustness is tested against noisy or adversarial examples to check resilience.
- Scalability and deployment feasibility of the best model are discussed based on memory usage and inference time.

## Step 8: Hyperparameter Tuning For Best Model

- We fine-tune the model’s settings—called hyperparameters—to help it learn better and make smarter predictions.
- Hyperparameters helps the model learn better patterns from attack data, leading to higher accuracy in detecting cyber threats.
- Hyperparameter tuning makes the final model more reliable, robust, and ready for real-time intrusion detection systems in real-world environments.

## Step 9: Final Results And Recommendations

- After evaluating seven machine learning models on attack-specific data from the CIC-IDS2017 dataset, the Random Forest Classifier consistently delivered the best overall.
- Techniques like SMOTE oversampling and feature scaling helped improve the model’s ability to handle class imbalance and generalize better.
- Finally, we suggest the best approach moving forward—whether it’s using this model, collecting more data, or trying different techniques.
    
You can access the CIC-IDS2017 dataset [here](https://www.unb.ca/cic/datasets/ids-2017.html).

