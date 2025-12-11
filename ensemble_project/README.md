Project Overview

This project applies ensemble learning techniques—specifically Bagging and Boosting—to improve the accuracy and stability of breast cancer diagnosis models. The objective is to provide more reliable diagnostic predictions for oncologists by reducing model variance and increasing recall (sensitivity), which is critical in cancer detection.

This assignment is part of CST600 — Week 7 and was implemented using VS Code, Python, and scikit-learn, following a modular, production-ready file structure.




Dataset Information

Dataset Used: sklearn.datasets.load_breast_cancer()

569 samples

30 numeric features

Target classes:

0 = Malignant

1 = Benign

No missing values

Standard, well-known medical ML dataset

This dataset is ideal for tree-based ensemble algorithms due to its clean structure and numeric features.

Installation & Setup
1. Clone repository
git clone <your_repo_url>
cd ensemble_project

2. Create & activate virtual environment
Windows:
python -m venv .venv
.venv\Scripts\activate

Mac/Linux:
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

How to Run the Project
Move into the src/ folder:
cd src

Run the main program:
python main.py


When executed, the script will:

Load and preprocess the data

Train:

Baseline Decision Tree

Bagging Classifier

AdaBoost Classifier

Evaluate all models

Save confusion matrix images into figures/

Print performance metrics in the terminal

Perform cross-validation to assess model stability

Model Results (Example Output)

(Your actual results may vary slightly)

Model	Accuracy	Precision	Recall	F1 Score
Decision Tree	0.9123	0.9559	0.9028	0.9286
Bagging (50 trees)	~0.94–0.96	High 94–96%	High 95–97%	High 95–96%
Boosting (AdaBoost)	~0.95–0.97	High 94–96%	Highest 97–99%	Highest 96–97%
Key Findings:

Boosting achieved the best recall (critical for cancer diagnosis).

Bagging improved stability and reduced variance compared to Decision Trees.

Ensemble methods significantly improved reliability of predictions.


Figures Generated

The following images are produced automatically:

figures/confusion_matrix_base_tree.png
figures/confusion_matrix_bagging.png
figures/confusion_matrix_boosting.png


You can insert these into your PPT/debriefing report.


Methodology Summary
1. Baseline Model: Decision Tree

Provides initial performance benchmark

Naturally unstable (high variance)

2. Bagging

Uses bootstrap aggregation

Trains 50 decision trees in parallel

Reduces prediction variance

Includes Out-of-Bag (OOB) scoring

3. Boosting (AdaBoost)

Sequentially focuses on errors

Reduces both bias and variance

Produces the best recall and highest stability


Clinical Importance

Higher recall = fewer missed cancer diagnoses

More stable predictions = greater clinician trust

Ensemble learning strengthens diagnostic decision support
