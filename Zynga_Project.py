# -*- coding: utf-8 -*-
"""
Created on Wed May  2 13:32:20 2018

@author: prita
"""

# Importing the libraries
import sys
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix

## Step 1 - Data collection : Download the data
user = pd.read_csv("C:\PritamData\Zynga\Website_VIP_User_data_10000.csv")

#Removing ID column as its not needed
user = user.iloc[0:10001 , 1:11]
        
## Step 2: Data exploration and preparation : Exploring and preparing the data ---- 
print("\nFew rows from dataset\n", user.head(10))
print("\nNo of rows and columns\n", user.shape)               ## No of rows and columns
print("\nDatatypes \n", user.dtypes)              ## Structure
print("\nLenght of Dataset : ", len(user.IsVIP_500))
   
X = user.iloc[:,1:10]
y = user.iloc[:,0]
print("\nCount of IsVIP_500\n",user.groupby('IsVIP_500').size())       ## count
print("\nProportion of IsVIP_500\n",y.value_counts() / 10001 * 100)    
y.value_counts().plot(kind='bar')   

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("\nSize of Training Dataset\n",X_train.shape)     ## count for X_train
print("\nSize of Testing Dataset\n",X_test.shape)     ## count for X_test

print("\nSize of Training Dataset\n",y_train.count())       ## count for y_train
print("\nSize of Testing Dataset\n",y_test.count()) 

print("\nSize of IsVIP_500 - train\n",y_train.value_counts())       ## count
print("\nSize of IsVIP_500 - test\n",y_test.value_counts()) 

print("\nProportion of IsVIP_500 - train\n",y_train.value_counts() / 7500 * 100)       ## proportion
print("\nProportion of IsVIP_500 - test\n",y_test.value_counts() / 2501 * 100) 

### Univariate Analysis

print("\nSummary of Numeric Variables\n",X.describe(include=[np.number]))       
### Describe() also helps to find missing values

###  Data Munging - Data Cleaning
##  Options- 1. Tremove rows with missing data from your dataset Or
###          2. Impute missing values with mean values in your dataset.
print("\nCheck any Null values :\n", X.isnull().sum())
print("\nCheck if all Finite values :\n", np.isfinite(X).sum())
print("\nCheck any NaN values :\n", np.isnan(X).sum())
X=X.replace([np.inf, -np.inf], 0)         ## Code to replace any infite number with NaN
X=X.replace(np.nan, 0)  
print("\nCheck any Null values :\n", X.isnull().sum())
print("\nCheck if all Finite values :\n", np.isfinite(X).sum())
print("\nCheck any NaN values :\n", np.isnan(X).sum())
Print("\nNull using Lambda", X.apply(lambda x: sum(x.isnull()),axis=0)) 

### Code to Find Null Values
#print("\nCheck any Null values : ", X.shape[0] - X.dropna().shape[0])
#print("\nCheck any Null values : ",X.isnull().values.ravel().sum())
# %timeit sum(map(any, X.apply(pd.isnull)))
## Code to fill null values - replacement by mean
#sdf['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)


### Outliers
### Visualizations Fo Find Outliers
#Draw a box plot for payment_7_day variable
#Do you suspect any outliers in payment_7_day ?
#Get relevant percentiles and see their distribution.
payment_7_day = X.iloc[:,0]
#%matplotlib inline 
plt.boxplot(payment_7_day)
payment_7_day.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.hist(payment_7_day, alpha=0.5, bins=50 )
###df.boxplot(column='ApplicantIncome', by = 'Education')  !!!!cCheck out
# Alternate way
#payment_7_day.plot(kind='hist', alpha=0.5, bins=20)
#payment_7_day.plot.box()

print("\nFrequency of Days of Customer Login\n",X.iloc[:,1].value_counts()) 
X_train.iloc[:,1].value_counts().plot(kind='bar')

No_of_active_days = X.iloc[:,2]
No_of_active_days.plot(kind='hist', alpha=0.5, bins=20)
No_of_active_days.plot.box()
##Here we observe that there are few extreme values. 
##This is also the reason why 50 bins are required to depict the distribution clearly.
###This confirms the presence of a lot of outliers/extreme values. 

total_trans = X.iloc[:,3]
print("\nFrequency of Transactions \n",total_trans.value_counts()) 
total_trans.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
total_trans.value_counts().plot(kind='bar')


product_like_rate = X.iloc[:,6]
plt.boxplot(product_like_rate)
product_like_rate.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.hist(product_like_rate, alpha=0.5, bins=20 )

## I figured out the 40 rows with -inf values for product_like_rate are some errors 
##  so decided to replace them with 0 for now to get the classfier working
## There are few products whoes rate is 19% which is not possible as likes/viewed can never be > 1
## so they are again some kind of errors, There are total 10% of such errors, we can not delete thoes rows

# BiVariate Analysis
scatter_matrix(X,  diagonal='kde')
#alpha=0.2 ,figsize=(6, 6),

plt.scatter(payment_7_day, y)
plt.xlabel("payments")
plt.ylabel("VIP")

plt.scatter(total_trans, y)
plt.xlabel("total transactions")
plt.ylabel("VIP")

plt.scatter(product_like_rate, y)
plt.xlabel("product like rate")
plt.ylabel("VIP")

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

print("\nSize of Training Dataset\n",len(y_train))       ## count for y_train
print("\nSize of Testing Dataset\n",len(y_test))       ## count for y_test
print("\nProportion of default - train\n",y_train.value_counts() / 8000 * 100)       ## count
print("\nProportion of default - test\n",y_test.value_counts() / 2001 * 100)   

## Step 3: Model Training:  Training a model on the data ----
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)
print("Classifier :", classifier)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = classifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: ",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", classifier.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", classifier.score(X_test, y_test, sample_weight=None))

## Step 5: Model Improvement : Improving model performance ----

### Using criterion='gini' and min_samples_leaf=5 and max_depth=3
print("\nModel Improvement Using criterion='gini' and min_samples_leaf=5 and max_depth=3")
giniclassifier = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=5, min_samples_split=4,
            min_weight_fraction_leaf=0.0, presort=False, random_state=100,
            splitter='best')
giniclassifier.fit(X_train, y_train)
print("Classifier :", giniclassifier)
t0=time()
y_pred = giniclassifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: ",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", giniclassifier.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", giniclassifier.score(X_test, y_test, sample_weight=None))


### Using criterion='entropy' and min_samples_leaf=5 and max_depth=3
print("\nModel Improvement Using criterion='entropy' and min_samples_leaf=5 and max_depth=3")
entropyclassifier = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=5, min_samples_split=4,
            min_weight_fraction_leaf=0.0, presort=False, random_state=100,
            splitter='best')
entropyclassifier.fit(X_train, y_train)
print("Classifier :", entropyclassifier)
t0=time()
y_pred = entropyclassifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: ",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", entropyclassifier.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", entropyclassifier.score(X_test, y_test, sample_weight=None))


### Using RandomForestClassifier and criterion='entropy'
ranforestclassifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=4, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=5, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
ranforestclassifier.fit(X_train, y_train)
print("Random Forest Classifier :", ranforestclassifier)
t0=time()
y_pred = ranforestclassifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: ",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", ranforestclassifier.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", ranforestclassifier.score(X_test, y_test, sample_weight=None))
scores = cross_val_score(ranforestclassifier, X, y)
print("Score is ", scores.mean())                             


extratreesclassifier = ExtraTreesClassifier(n_estimators=10, criterion='entropy', 
        max_depth=4, min_samples_split=2, min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, max_features='auto', 
        max_leaf_nodes=None, min_impurity_decrease=0.0, 
        min_impurity_split=None, bootstrap=False, oob_score=False, 
        n_jobs=1, random_state=None, verbose=0, warm_start=False, 
        class_weight=None)
extratreesclassifier.fit(X_train, y_train)
print("Random Forest Classifier :", extratreesclassifier)
t0=time()
y_pred = extratreesclassifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: ",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", extratreesclassifier.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", extratreesclassifier.score(X_test, y_test, sample_weight=None))
scores = cross_val_score(extratreesclassifier, X, y)
print("Score is ", scores.mean())            


gradientboostingclassifier = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,
    max_depth=4, random_state=0)
gradientboostingclassifier.fit(X_train, y_train)
print("Gradient Boosting Classifier :", gradientboostingclassifier)
t0=time()
y_pred = gradientboostingclassifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: ",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", gradientboostingclassifier.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", gradientboostingclassifier.score(X_test, y_test, sample_weight=None))
scores = cross_val_score(gradientboostingclassifier, X, y)
print("Score is ", scores.mean())            




#_, bp = pd.DataFrame.boxplot(df, return_type='both')
#outliers = [flier.get_ydata() for flier in bp["fliers"]]
#boxes = [box.get_ydata() for box in bp["boxes"]]
#medians = [median.get_ydata() for median in bp["medians"]]
#whiskers = [whiskers.get_ydata() for whiskers in bp["whiskers"]]



###Categorical variable analysis

#temp1 = df['Credit_History'].value_counts(ascending=True)
#temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
#print 'Frequency Table for Credit History:' 
#print temp1
#print '\nProbility of getting loan for each Credit History class:' 
#print temp2