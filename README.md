# Titanic Survival Prediction - Classification

Predictions will be made on survival outcomes based on the Titanic dataset.

The target variable, whether a passenger survived or not, is a binary outcome (0 or 1), making it suitable for classification modeling. Therefore, classification algorithms are appropriate for this task.

## Data
The dataset consists of two parts: the training set (ttrain.csv) and the test set (ttest.csv). The training set contains information about a subset of passengers along with their survival status, while the test set contains information about another subset of passengers without survival status.

<h3>Data Dictionary</h3>
<table>
  <tr>
    <th>Variable</th>
    <th>Definition</th>
    <th>Key</th>
  </tr>
  <tr>
    <td>Survival</td>
    <td>Survival</td>
    <td>0 = No, 1 = Yes</td>
  </tr>
  <tr>
    <td>Pclass</td>
    <td>Ticket class</td>
    <td>1 = 1st, 2 = 2nd, 3 = 3rd</td>
  </tr>
  <tr>
    <td>Sex</td>
    <td>Gender of the passenger</td>
    <td></td>
  </tr>
  <tr>
    <td>Age</td>
    <td>Age in years</td>
    <td></td>
  </tr>
  <tr>
    <td>SibSp</td>
    <td>Number of siblings / spouses aboard the Titanic</td>
    <td></td>
  </tr>
  <tr>
    <td>Parch</td>
    <td>Number of parents / children aboard the Titanic</td>
    <td></td>
  </tr>
  <tr>
    <td>Ticket</td>
    <td>Ticket number</td>
    <td></td>
  </tr>
  <tr>
    <td>Fare</td>
    <td>Passenger fare</td>
    <td></td>
  </tr>
  <tr>
    <td>Cabin</td>
    <td>Cabin number</td>
    <td></td>
  </tr>
  <tr>
    <td>Embarked</td>
    <td>Port of Embarkation</td>
    <td>C = Cherbourg, Q = Queenstown, S = Southampton</td>
  </tr>
</table>


### Variable Notes
* Pclass: A proxy for socio-economic status (SES)
  * 1st = Upper
  * 2nd = Middle
  * 3rd = Lower
* Age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
* SibSp: The dataset defines family relations in this way...
  * Sibling = brother, sister, stepbrother, stepsister
  * Spouse = husband, wife (mistresses and fiancés were ignored)
* Parch: The dataset defines family relations in this way...
  * Parent = mother, father
  * Child = daughter, son, stepdaughter, stepson
      * Some children travelled only with a nanny, therefore parch=0 for them.

## Exploratory Data Analysis (EDA)
Before delving into modeling, an exploratory data analysis (EDA) was conducted to gain insights into the Titanic dataset. This involved examining the dataset's structure, checking for missing values, and exploring relationships between variables. Key findings from the EDA included the presence of 1309 entries with various features. Missing values were identified in columns such as Age, Fare, Cabin, and Embarked. These insights informed data preprocessing steps and laid the foundation for subsequent modeling.

## Modelling
Several machine learning models were developed using popular algorithms such as Gaussian Naive Bayes, Bernoulli Naive Bayes, Random Forest Classifier, and Gradient Boosting Classifier. After fitting the models, predictions were generated for the test set. 

## Results
* I have submitted the predictions generated by different classifiers to the Titanic Kaggle competition: 
https://www.kaggle.com/c/titanic. Here are the scores obtained:

### Model Performance:

1) Score: 0.77511 | GradientBoostingClassifier

2) Score: 0.77033 | BernoulliNB

3) Score: 0.75837 | GaussianNB

4) Score: 0.74162 | RandomForestClassifier

These scores may vary slightly due to factors such as randomness in the algorithms. 
Among the algorithms used, RandomForestClassifier and GradientBoostingClassifier can exhibit randomness.

### Required Packages
Here's a summarized list of required packages and their descriptions:

* pandas: Data manipulation and analysis library for Python.

* seaborn: Statistical data visualization library based on matplotlib, simplifying the creation of complex and attractive plots.

* scikit-learn: Machine learning library in Python offering tools for data mining, analysis, and model evaluation.
