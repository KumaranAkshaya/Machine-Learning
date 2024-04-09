# Machine-Learning
A project focusing on credit card fraud detection using machine learning techniques.
# Abstract
As the number of credit card transactions keep growing and represent an increasing share of the 
European payment system. Leading to several stolen account numbers and subsequent losses to 
banks, also people believed that credit card transaction fraud is a growing threat with severe 
implications for the financial industry. Data mining (Machine Learning) plays a crucial role in 
detecting credit card fraud in both online and offline transactions. Credit card fraud detection 
which is a data mining problem becomes challenging for two main reasons. First, the 
characteristics of normal and fraudulent behavior are continually changing, and second, the credit 
card fraud dataset is highly asymmetric. The performance of fraud detection in credit card 
transactions is greatly affected by the sampling method of the dataset and the choice of variables 
and the detection techniques used. This paper investigates the performance of linear regression 
(LR), logistic regression (LR), k-nearest neighbor (KNN), Support Vector Machine (SVM), 
Decision Tree (DT), ANN, MLP, Random Forest, XG Booster and Naïve Bayes on credit card 
fraud data. The dataset of credit card transactions obtained from Kaggle containing 284,807
transactions. A mixture of under-sampling and oversampling techniques applied to the unbalanced 
data. The five strategies used to the raw and preprocessed data, respectively. This work 
implemented in Python.
# Acknowledgement
We would like sincerely to thank the author from the Kaggle platform which offers the dataset.
The dataset has been collected and analyzed during a research collaboration of Worldline and the 
Machine Learning Group (http://mlg.ulb.ac.be) of ULB on big data mining and fraud detection. 
More details on current and past projects on related topics are available on 
https://www.researchgate.net/project/Fraud-detection-5 and the page of the Defeat Fraud project.
# Dataset:
The dataset contains transactions made by credit cards in September 2013 by European 
cardholders. This dataset presents transactions that occurred in two days, where we have 
492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class 
(frauds) account for 0.172% of all transactions. It contains only numerical input variables 
which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, 
we cannot provide the original features and more background information about the data. 
Features V1, V2, … V28 are the principal components obtained with PCA, the only 
features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 
'Time' contains the seconds elapsed between each transaction and the first transaction in 
the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for 
example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it 
takes value 1 in case of fraud and 0 otherwise. Given the class imbalance ratio, we 
recommend measuring the accuracy using the Area Under the Precision-Recall Curve 
(AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
# dataset link from kaggle
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Steps:
# Loading the data: 
Load the credit card transaction data from a CSV file using pandas.
# Data exploration:
Display the first and last 5 rows of the dataset. Check basic information 
about the dataset (data types, non-null counts, etc.)
# Data preprocessing:
Separate the data into legitimate (non-fraudulent) and fraud 
transactions. Display statistical measures for both classes.
# Under-sampling: 
Create a new dataset with a balanced distribution of both legitimate and 
fraudulent transactions.
# Data visualization:
Plot the distribution of classes in the new dataset. Visualize the 
distribution of 'Amount' and 'Time' features for both classes.
# Splitting the Data: 
Split the dataset into features (X) and target (Y). Further split the data 
into training and testing sets.
# Model training: Train a machine learning model using the training dataset. Use the trained 
model to make predictions on the test dataset. Assess the model's performance using 
various metrics such as precision, recall, F1-score, and AUC-ROC. 
# Steps under model training
# 1.Precision:
Precision measures the accuracy of positive predictions made by the model. In our case, it 
represents the proportion of correctly predicted spam emails out of all emails predicted as spam. 
Using the confusion matrix, we can calculate precision as:
Precision = True Positives / (True Positives + False Positives)
# 2. Recall:
Recall, also known as sensitivity or true positive rate, measures the ability of the model to correctly 
identify positive instances. It represents the proportion of correctly predicted spam emails out of 
all actual spam emails. Using the confusion matrix, we can calculate recall as:
 Recall = True Positives / (True Positives + False Negatives)
# 3. F1 Score:
The F1 score combines precision and recall into a single metric. It provides a balanced measure 
that considers both false positives and false negatives. The F1 score can be calculated as the 
harmonic mean of precision and recall:
 F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
# 4. AUC ROC:
AUC ROC measures the performance of a binary classification model by plotting the true positive 
rate against the false positive rate at various classification thresholds. It quantifies the model's 
ability to distinguish between spam and legitimate emails. The higher the AUC ROC value, the 
better the model's discrimination. For our spam detection model, let's assume we obtained an AUC 
ROC score of 0.95.
# Checking the accuracy: Calculate the overall accuracy of the model by dividing the 
number of correct predictions by the total number of predictions. Accuracy = (Number of 
Correct Predictions) / (Total Number of Predictions).
# Experimental results and discussion:
From the results, we observe that the MLP Classifier has the highest accuracy on the testing data
set (0.9492), followed closely by Logistic Regression (0.9441). However, it's essential to conside
r other metrics like precision, recall, and F1-score to get a comprehensive understanding of mode
l performance.
While MLP Classifier shows the highest accuracy, it has significantly lower recall compared to o
ther models, indicating that it may not be the best choice if detecting all instances of fraud is cruc
ial. Logistic Regression, on the other hand, provides a good balance between precision, recall, an
d accuracy.
Overall, Logistic Regression seems to be a strong candidate for credit card fraud detection in this 
scenario, offering a good balance between different evaluation metrics. However, further fine-tun
ing and testing on larger datasets may be necessary to confirm its effectiveness in real-world appl
ications



