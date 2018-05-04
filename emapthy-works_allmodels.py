
# coding: utf-8

import pandas as pd
import numpy as np
import statistics
from sklearn.preprocessing import LabelEncoder
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_validation import KFold, train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ### Reading the CSV file into a Pandas DataFrame and calculating aggregate functions for the Target column

df = pd.read_csv('responses.csv')

df['Empathy'].describe()


# ### Find and drop the rows with NaN values in Empathy column


df2 = df['Empathy'].isnull().sum()
print("Number of rows with NaN values to be dropped from Empathy column:",df2)

df.dropna(subset = ["Empathy"], inplace=True)


ax = plt.axes()
ax.set_title("Count of all classes for Empathy")
qw = sns.countplot(y = 'Empathy', data = df,ax = ax)


# ### Find all the integer and the categorical columns

int_columns = []
categorical_columns = []
for i in df.columns:
    if (df[i].dtype!='object'):
        int_columns.append(i)
        
        
for i in df.columns:
    if (df[i].dtype=='object'):
        categorical_columns.append(i)
print ('Number of integer features:',len(int_columns))
print('Number of categorical features:',len(categorical_columns))


print("\n For categorical columns: Filling NaN values with mode of Column and Label Encode them")

le = LabelEncoder()
for i in categorical_columns:
    abc = np.array(df[i])
    x = statistics.mode(abc)
    xyz = df[i].fillna(x)
    aqws = le.fit_transform(xyz)
    df[i] = aqws




print("\n Filling all the Null values in integer columns with mean values")

means = df.mean(axis=0)
df.fillna(means, inplace=True)


print("\nFind the Correlation of each column with Target column")

def correlation(target, df):
    df1 = copy.deepcopy(df)
    d1 = {}
    for i in df.columns:
        corref = (np.corrcoef(df[i],df[target])[0,1])
        d1[i] = corref
    return d1


cooer = correlation('Empathy',df)
#print("Correlation value dictionary (Key: column name, value = correlation_value)\n",cooer)


##Obtain the Top N feature columns based on correlation values


from collections import OrderedDict
d = OrderedDict(sorted(cooer.items(), key=lambda t: t[1],reverse=True))
i=0
lad = []
count=0
for key,value in d.items():
    lad.append(key)
    
lad1 = lad[1:11]
lad2 = lad[1:21]
lad = lad1
lad_main = lad2 

cooer1 = copy.deepcopy(d)
cooer1.pop('Empathy')
fig, ax = plt.subplots(figsize= (10,30))
ax.set_title("Correlation coefficient of the columns with Empathy")
ax.set_ylabel("Correlation coefficients")
sns.barplot(x=list(cooer1.values()), y=list(cooer1.keys()),  ax = ax,color = 'r')


# ### Create a dataframe with just the relevant features in one and another with just target label for Training and Testing
X = copy.deepcopy(df[lad])
X1 = copy.deepcopy(df[lad_main])
Y = df["Empathy"]


# # Baseline Model
A = df.drop(['Empathy'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(A, Y, test_size=.3, random_state=100)

clf = LinearSVC(multi_class = 'ovr',random_state = 42)
clf.fit(x_train,y_train)


# **Accuracy and Classifiation Report**

pred = clf.predict(x_test)
acc3 = np.mean(pred==y_test)
print("\n\nBaseline Model using Linear SVM\n")
print("Basline accuracy:",acc3,"\n") 
print(classification_report(pred,y_test))


# ### *Perform Train Test split of 30% for training and testing*

# # Proposed model with Correlations only

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=100)


# **Linear SVM classifier with one vs rest method for multi-class classification**

clf = LinearSVC(multi_class = 'ovr',random_state = 100)
clf.fit(x_train,y_train)


# **Accuracy and Classification report**


pred = clf.predict(x_test)
acc = np.mean(pred==y_test)
print("\n\nProposed Model using SVM with correlations:\n")
print("accuracy of proposed model:",acc) 
print("\nclassification report:\n",classification_report(pred,y_test))


# # Proposed model with standard scaling


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=100)


# ### Standard Scaling
#  - Standarizes the features by modifying the values such that their mean is 0 and variance is 1


scaler = StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# **Linear SVM classifier with One VS Rest method for multi-class classification**


clf = LinearSVC(multi_class = 'ovr',random_state = 42)
clf.fit(x_train,y_train)


# **Accuracy and Classification Report**

pred = clf.predict(x_test)
acc1 = np.mean(pred==y_test)
print("\n\nProposed Model using SVM with correlations and Standard Scaling:\n")
print("accuracy of proposed model:",acc1)
print("\nclassification report:\n",classification_report(pred,y_test))


# # Proposed model with GridsearchCV for hyperparameter tuning

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=100)


# ### GridsearchCV
#    - SVM classifier as base estimator
#    - Paramters tuned include Kernel and penalty C
#    - 5 fold cross validation for tuning


clf = SVC(random_state = 21)

KF = KFold(len(x_train), n_folds=5)
param_grid = {'kernel':('linear', 'rbf'),'C':[0.0001,0.003,.001,.01,.03,.1,.3,1,3,10]}
grsearch = GridSearchCV(clf, param_grid=param_grid, cv=KF)
grsearch.fit(x_train,y_train)
clf = SVC(C=grsearch.best_params_['C'],kernel = grsearch.best_params_['kernel'])
clf.fit(x_train,y_train)


# **Accuracy and Classification Report**

pred = clf.predict(x_test)
acc2 = np.mean(pred==y_test)
print("\n\nProposed Model using SVM with correlations and GridsearchCV:\n")
print("accuracy of proposed model:",acc2)
print("\nclassification report:\n",classification_report(pred,y_test))


# # Proposed model with Top 20 Features

x_train, x_test, y_train, y_test = train_test_split(X1, Y, test_size=.3, random_state=100)


# ### Standard Scaling and GridsearchCV hyperparameter tuning

scaler = StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

clf = SVC(random_state = 7)

KF = KFold(len(x_train), n_folds=5)
param_grid = {'kernel':('linear', 'rbf'),'C':[0.0001,0.003,.001,.01,.03,.1,.3,1,2,3,5,10]}
grsearch = GridSearchCV(clf, param_grid=param_grid, cv=KF)
grsearch.fit(x_train,y_train)
clf = SVC(C=grsearch.best_params_['C'],kernel = grsearch.best_params_['kernel'])
clf.fit(x_train,y_train)


# **Accuracy and Classification_report**


pred = clf.predict(x_test)
acc4 = np.mean(pred==y_test)
print("\n\nProposed Model Using SVM with Top 20 features")
print("\naccuracy of combination of all methods in proposed model with Top 20 features:",acc4,"\n")
print("classification report:\n",classification_report(pred,y_test))


#  - **It can be noticed that there is slight improvement in the fscores of the minority classes( class 1 and 2). The model learns the features better**
#  - **Increasing the number of features just helps us learn minority classes by a fraction**
#  - **Enabling another statistical test like t-test or correlation of columns with each class would help us learn the minority class relations better and achieve higher fscores**


d={'basline':acc3,'correlation':acc,'standardscaling':acc1,'gridsearchcv':acc2,'top20features':acc4}
ax.set_title("Accuracies for various models")
ax.set_ylabel("Accuracy")
sns.barplot(x=list(d.keys()), y=list(d.values()))

