import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec 
data  = pd.read_csv('creditcard.csv') 
fraud = data[data['Class'] == 1] 
valid = data[data['Class'] == 0] 
outlierFraction = len(fraud)/float(len(valid)) 
print(outlierFraction) 
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))
X = data.drop(['Class'], axis = 1) 
Y = data["Class"] 
print(X.shape) 
print(Y.shape) 
xData = X.values 
yData = Y.values
from sklearn.model_selection import train_test_split 
xTrain, xTest, yTrain, yTest = train_test_split( 
     xData, yData, test_size = 0.2, random_state = 42) 

# Using Skicit-learn to split data into training and testing sets 
from sklearn.ensemble import IsolationForest
clf = IsolationForest(max_samples= len(xTrain) , contamination=outlierFraction).fit(xTrain)
yPred = clf.predict(xTrain)
yPred[yPred ==1] = 0
yPred[yPred ==-1] = 1

from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix 
n_outliers = len(fraud) 
n_errors = (yPred != yTrain).sum() 
print("The model used is ISOLATION Forest classifier") 

acc= accuracy_score(yTrain, yPred) 
print("The accuracy is {}".format(acc)) 


report = classification_report(yTrain,yPred)
print(report)

# printing the confusion matrix 
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(yTrain, yPred) 
plt.figure(figsize =(6,6)) 
sns.heatmap(conf_matrix, xticklabels = LABELS, 
			yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 
