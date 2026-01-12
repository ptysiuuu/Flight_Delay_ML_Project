# The Application of Machine Learning Methods in Flight Delay Problems

This research project explores the application of diverse machine learning methodologies to predict, quantify, and segment flight delays using a large-scale dataset of approximately 7 million domestic USA flight records. Developed as a group project for the Machine Learning course at **Universidad de Deusto** during an exchange in **Bilbao**, this study provides a deep dive into the stochastic nature of aviation and the efficacy of various algorithmic approaches in handling high-dimensional, imbalanced data.

## Project Overview

The research tackles the problem of flight delays through three distinct machine learning lenses:
* **Classification:** Predicting beforehand whether a flight would be delayed on arrival (defined as any arrival delay > 0 minutes) using only features known prior to departure.
* **Regression:** Quantifying the linear propagation of delays to determine how much of the arrival delay is simply a continuation of departure delay.
* **Clustering:** Applying unsupervised learning to segment delayed flights into profiles to understand underlying behaviors and similarities.

## Data Engineering & Pre-processing

Handling a dataset of nearly 7 million records required rigorous pre-processing and engineering to ensure data quality and model performance:
* **Data Cleaning:** The dataset was reduced from 7.08 million to 6.97 million records by handling missing values and ensuring no duplicate entries existed.
* **Feature Engineering:** Time-based variables were converted into cyclic functions (sine and cosine pairs) for features like month, day of the week, and departure time. This allowed the models to capture the periodic nature of time effectively.
* **Addressing Imbalance:** With roughly 36% of flights delayed and 64% on time, the methodology prioritized the maximization of recall for delayed flights, recognizing that failing to predict a delay is more costly than a false alarm.

## Methodological Competence

### Advanced Classification

The project evaluated Support Vector Machines (SVM), Naive Bayes, Random Forest, and XGBoost.
* **XGBoost Excellence:** Identified as the superior model, it balanced high recall (0.6054) with strong ROC AUC and PR AUC metrics.
* **High-Performance Computing:** Due to computational weight, the SVM model was trained on a GPU using the **RAPIDS suite** to accelerate the processing of non-linear decision boundaries.
* **Optimization:** Hyperparameters were tuned using **Optuna** and **Randomized SearchCV** with StratifiedKFold to preserve class distribution.

### Regression & Hypothesis Testing

The regression task was designed as an experiment to test the predictability of delay duration.
* **Linear vs. Non-Linear:** The study confirmed that arrival delay is almost entirely linearly correlated with departure delay, with an R-squared value of approximately 0.95.
* **Findings:** Simple linear models outperformed complex non-linear models like XGBoost Regressor when predicting minutes of delay, proving that delays are often stochastic operational events rather than predictable structural patterns.

### Unsupervised Clustering

Unsupervised learning was used to identify five distinct profiles of delayed flights, such as "WN/CLT Local Shuttles" and "DL/LAX Transcontinental".
* **Deep Learning Integration:** For Gaussian Mixture Models (GMM), a **PyTorch-based autoencoder** was developed to compress data into a latent space, ensuring the model received dense numerical vectors while reducing computational cost.
* **Root Cause Analysis:** Clustering revealed that in short-distance clusters, the primary delay reason was late aircraft arrival (cascade effect), whereas long-haul clusters were primarily affected by internal carrier operations.

## My Technical Competence & Skills

This project allowed me to develop and demonstrate several core competencies essential for a Machine Learning and Computer Science internship:

* **Scalable Data Science:** I gained extensive experience managing and modeling a massive dataset of 7 million records using tools like pandas and numpy. To handle the computational load, I utilized GPU-accelerated libraries and the RAPIDS suite, allowing me to process complex models like SVM on high-dimensional data.

* **Model Interpretability:** I applied advanced interpretability techniques, including **SHAP values** and **Accumulated Local Effects (ALE)** plots, to decode black-box models. Through this analysis, I identified that the time of day and specific airport locations are the most critical factors in predicting flight delays.

* **Experimental Rigor:** I designed structured experiments to test specific hypotheses, such as comparing the predictive power of structural features against the linear propagation of departure delays. I also carefully selected evaluation metrics like PR AUC and Recall to ensure the models addressed the real-world costs of missed delay predictions.

* **End-to-End ML Development:** I managed the entire machine learning lifecycle, from raw data pre-processing and cyclic feature engineering using sine and cosine transformations for time-based variables to advanced model deployment and rigorous statistical evaluation.

* **Deep Learning Integration:** I integrated deep learning components into unsupervised tasks by building a **PyTorch-based autoencoder**. This allowed me to compress high-dimensional data into a latent space, significantly improving the efficiency and interpretability of Gaussian Mixture clustering.

## Authors

Adam Szostek, Monika Jung, Salma Ennassar, and Nassima EL Garn
