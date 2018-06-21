# Machine Learning for Kawasaki Disease Diagnosis

Kawasaki Disease (KD) is a rare heart condition that affects children all over the world. We use Kawasaki Disease patient data to train various machine learning models to predict whether a given patient has Kawasaki Disease or is a febrile control (i.e. does not have the disease). More information on Kawasaki Disease can be found [here](https://www.mayoclinic.org/diseases-conditions/kawasaki-disease/symptoms-causes/syc-20354598).

### Usage Instructions:
1. `git clone` this repository; `cd` to repository directory
2. Create a symlink from KD-data Dropbox folder to `deep-learning-kd-diagnosis/data`
3. Create conda environment: `conda create -n kd`; `source activate kd`
4. Install requirements: `pip install -r requirements.txt`
5. Run experiments: `bash run_run_all.sh`

### Methods Evaluated:
* K-Nearest Neighbors (K-NN)
* Logistic Regression
* Support Vector Machine (SVM)
* Tree-Based Methods: Random Forest, XGBoost
* Deep Neural Network
* Ensemble (Voting/Bagging) Classifiers

### Evaluation Methodologies/Metrics:
* 5-Fold (Nested) Cross Validation for model selection and evaluation
* **Metrics**: Sensitivity, Specificity; ROC-AUC

### Dependencies:
* Python 3.x
* Numpy/Scipy
* Scikit-Learn
* Matplotlib
* Tensorflow
* XGBoost
* [Fancyimpute](https://github.com/iskandr/fancyimpute)
* [Scikit-Optimize](https://scikit-optimize.github.io/)
* [Imbalanced-Learn](https://github.com/scikit-learn-contrib/imbalanced-learn)
