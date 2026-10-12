# ML-_mini-Project-# Radio Pulsar Candidate Classification Using Machine Learning

## Overview

Radio pulsar surveys generate thousands of candidate signals, but only a small fraction correspond to genuine pulsars. Manually examining these candidates is time-consuming and makes automated classification an important application of machine learning in astronomy.

This project reproduces and extends the research paper **"Application of Machine Learning Methods to Identify and Categorize Radio Pulsar Signal Candidates."** It uses supervised and unsupervised machine learning techniques to distinguish genuine pulsar signals from non-pulsar signals caused by noise and Radio Frequency Interference (RFI).

The project follows a two-part approach:
- **Reproduction:** Implement the models and reproduce the experiments reported in the reference paper.
- **Extension:** Develop an additional machine learning model and evaluate its performance against the reference approaches.

## Problem Statement

Develop a machine learning pipeline that classifies radio pulsar candidates using statistical features extracted from their signals. The system should identify potential pulsars while minimizing missed positive candidates and explore the grouping of pulsar candidates through unsupervised learning.

## Dataset

The project uses the **HTRU2 dataset**, available through the UCI Machine Learning Repository.

| Property | Description |
|---|---|
| Total samples | 17,898 |
| Genuine pulsar samples | 1,639 |
| Non-pulsar samples | 16,259 |
| Input features | 8 statistical features |
| Target variable | 1 = Pulsar, 0 = Non-pulsar |
| Data type | Numerical |

### Features

The dataset contains eight continuous statistical features extracted from two signal representations:

**Integrated Pulse Profile**
1. Mean
2. Standard deviation
3. Excess kurtosis
4. Skewness

**DM-SNR Curve**
5. Mean
6. Standard deviation
7. Excess kurtosis
8. Skewness

These features describe the statistical characteristics of the observed radio signals and are used to train the classification models.

**Dataset source:** [UCI Machine Learning Repository – HTRU2](https://archive.ics.uci.edu/dataset/372/htru2)

## Methodology

The project is organized into the following stages:

1. **Data loading:** Load the HTRU2 dataset and inspect its features, target distribution, and data quality.
2. **Preprocessing:** Prepare the feature matrix and target labels. Apply feature scaling where required by the algorithm.
3. **Class imbalance handling:** Upsample positive pulsar examples in the training data only, leaving validation and test data untouched.
4. **Supervised learning:** Train classification models to distinguish pulsars from non-pulsars.
5. **Unsupervised learning:** Cluster genuine pulsar candidates to investigate similarities in their statistical features.
6. **Dimensionality reduction:** Use Principal Component Analysis (PCA) to visualize high-dimensional data and clustering results in two dimensions.
7. **Model evaluation:** Compare classification metrics and assess clustering quality.
8. **Model extension:** Evaluate an additional K-Nearest Neighbors (KNN) classifier against the baseline approach.

## Models Implemented

### 1. Gaussian Discriminant Analysis (GDA)

GDA serves as the baseline supervised learning model. It models the distribution of feature values for each class and uses the resulting probability estimates to classify pulsar candidates.

### 2. Random Forest — Pending Implementation

The reference paper uses Random Forest as its primary supervised learning approach. The reported configuration includes:

- Number of trees: 200
- Minimum samples split: 5
- Maximum features: 4
- Maximum depth: 110

The reference also evaluates a decision threshold of 0.2 to improve recall. Implementation and evaluation in this project remain pending.

### 3. K-Means Clustering

K-Means is used as the baseline unsupervised learning algorithm. It groups the positive-labeled pulsar examples into three clusters based on their feature similarity.

### 4. Self-Organizing Map (SOM) — Pending Implementation

The reference paper uses a Self-Organizing Map to cluster pulsar candidates through a neural-network-based, unsupervised learning approach.

The reference configuration uses a 5 × 5 output map and 100 iterations. Implementation and evaluation in this project remain pending.

### 5. Principal Component Analysis (PCA)

PCA reduces the feature space to two principal components for visualization. It is applied after clustering and is used to display cluster distributions, rather than as a preprocessing step for the clustering algorithms.

### 6. K-Nearest Neighbors (KNN) — Additional Model

KNN is the additional model introduced as the project extension. It classifies a candidate based on the labels of its nearest neighbors in the feature space.

Feature scaling is applied before KNN because distance-based algorithms are sensitive to differences in feature magnitudes.

The number of neighbors was selected through cross-validation, with **k = 3** chosen for the final model.

## Experimental Results

### Current Implementation Results

The following results have been obtained from the current implementation.

| Model | Evaluation | Accuracy | Recall |
|---|---|---:|---:|
| GDA | Test set | 97.85% | 89.63% |
| KNN (k = 3) | Test set | 98.02% | 85.67% |
| K-Means | Clustering | N/A | N/A |

K-Means achieved a **Silhouette Coefficient of 0.4413**, which measures the separation and cohesion of the resulting clusters.

For the additional KNN model, the other test-set metrics are:

| Metric | Result |
|---|---:|
| Precision | 92.13% |
| F1-score | 88.78% |
| Accuracy | 98.02% |
| Recall | 85.67% |

The KNN confusion matrix is:

| | Predicted Non-pulsar | Predicted Pulsar |
|---|---:|---:|
| Actual Non-pulsar | 3,228 | 24 |
| Actual Pulsar | 47 | 281 |

### Reference Paper Results

The reference paper reports the following test-set results:

| Model | Accuracy | Recall |
|---|---:|---:|
| Gaussian Discriminant Analysis | 96.28% | 83.43% |
| Random Forest (threshold = 0.5) | 97.93% | 83.13% |
| Random Forest (threshold = 0.2) | 97.93% | 88.00% |

The paper also reports a Silhouette Coefficient of approximately 0.44 for both K-Means and SOM.

**Note:** Results can vary with data splitting, random seeds, preprocessing, hyperparameters, and library versions. Final reproduction comparisons should use consistent evaluation protocols and clearly distinguish published results from results obtained in this implementation.

## Key Findings

- The dataset is imbalanced, making recall an important metric alongside accuracy.
- GDA provides a useful baseline for binary pulsar classification.
- KNN achieves 98.02% test accuracy in the current implementation, with an 85.67% recall.
- The choice of decision threshold can change the trade-off between detecting genuine pulsars and producing false positives.
- K-Means produces three clusters, but a Silhouette Coefficient of approximately 0.44 indicates that the clusters are not strongly separated.
- PCA helps visualize the data but does not, by itself, improve the clustering model.
- Random Forest and SOM still need to be implemented and evaluated before the reproduction stage is complete.

## Technologies Used

- Python
- Google Colab
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow (planned for SOM implementation)

## Getting Started

### Prerequisites

- Python 3.x or Google Colab
- Jupyter Notebook
- The HTRU2 dataset
- The Python libraries listed above

### Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Open the project notebook in Jupyter Notebook or upload it to Google Colab.

Download the HTRU2 dataset from the UCI Machine Learning Repository and ensure the notebook points to the correct dataset path.

Run the notebook cells sequentially to execute preprocessing, model training, evaluation, and visualization.

## Project Structure

```text
radio-pulsar-classification/
│
├── README.md
├── notebooks/
│   └── pulsar_classification.ipynb
├── data/
│   └── htru2.csv
└── results/
    └── evaluation_plots/
```

*The structure above is a suggested organization. Adjust the filenames and directories to match the actual GitHub repository.*

## Future Work

- Complete the Random Forest implementation and reproduce the paper's threshold-based evaluation.
- Implement the Self-Organizing Map and compare its clusters with K-Means.
- Perform a consistent comparison between the published results and the reproduced results.
- Evaluate additional metrics, including precision, F1-score, and ROC-AUC.
- Investigate feature importance and the effect of different preprocessing strategies.
- Explore additional classification algorithms if needed.

## References

1. Debesai, S., Gutierrez, C., and Koyluoglu, N. U., *Application of Machine Learning Methods to Identify and Categorize Radio Pulsar Signal Candidates*, 2020.
2. Lyon, R. J., et al., *Fifty Years of Pulsar Candidate Selection: From Simple Filters to a New Principled Real-Time Classification Approach*, Monthly Notices of the Royal Astronomical Society, vol. 459, no. 1, pp. 1104–1123. DOI: [10.1093/mnras/stw656](https://doi.org/10.1093/mnras/stw656).
3. Lyon, R. J., *HTRU2 Dataset*, UCI Machine Learning Repository. [Dataset link](https://archive.ics.uci.edu/dataset/372/htru2).

---

**Project objective:** Reproduce the reference paper's machine learning experiments, evaluate their performance, and extend the study with an additional KNN classifier for radio pulsar candidate identification.
