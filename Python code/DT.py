
#DATA ingest

import pandas as pd
import os
os.getcwd()
os.chdir("C://Users//tanay//desktop//Data.Science.Portfolio//Diabetes")
data = pd.read_csv("diabetes.csv")
pd.set_option("display.max_rows", None, "display.max_columns", None) 

#DATA EXPLORATION


import seaborn as sns
correlation = data.corr()
sns.heatmap(correlation,
            xticklabels = data.columns,
            yticklabels = data.columns)


#Class distribution of response variable:
data.Outcome.value_counts()

data.hist(figsize=(9,9)) # shows frequencies of each feature, depicting any imbalances
#Examine impact of "Age" on outcome:

#Diabetes per age group
ct = pd.crosstab(data.Age, data.Outcome)
ct.plot(kind = "bar", rot =0, figsize = (10,10))

data.Age.describe() #get stats for age
data[data.Age < 29].shape[0] # how many are below age 29
data[data.Age > 29].shape[0] # how many are above age 29

# import matplotlib.pyplot as plt
# plt.grid(True)
# plt.xlabel("Age")
# plt.ylabel("% diabetic")
# plt.plot(sorted(data.Age.unique()), ct["% diabetic"])
# data.sort_values("Age") # sort dataframe by ascending age values

"""
shows us that between 20 and 30, very low percentage of diabetics
between 30 and 60, roughy 40 - 55 % on average which means age alone has limiting effect on outcome
"""

data[data.Age.between(21,30)].head(20) # ages from 21 to 30 inclusive 

# DATA CLEANING

#Check for null or na values 

data.isnull().sum()
data.isna().sum()


data.Outcome.plot()


# DATA OUTLIERS  AND CLEANING
data_clean = data.copy()

# identify qty of zeros in each feature for outlier evaluation 

for col in data_clean.columns:
    print(f"{col}: {data_clean[data_clean[col] == 0].shape[0]}/{data_clean[col].shape[0]} entries are zero")



# Glucose : 5 entries 
# BP : 35 entries
# BMI: 11 entries
# Skin thickness = 227 entries -  too many to remove and doesnt make sense to fill w/ the median 
# Insulin = 374 entries  - too many to remove and doesnt make sense to fill w/ the median 

# Remove all zeros from dataframe
data_clean = data_clean [ (data.Glucose != 0) & (data.BloodPressure != 0) & (data.BMI != 0)  ]





# Create X and Y

X = data_clean.iloc[:, :-1]
Y = data_clean.Outcome

# Split dataset into training & test sets

from sklearn.model_selection import train_test_split

# 75/25 split using stratification to ensure equal class representation in splits
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state  = 0,
                                                    stratify = Y, 
                                                    test_size=0.25, train_size=0.75)

# verify equal class representation
y_train.value_counts()[0]/y_train.value_counts()[1] # 1.9037
y_test.value_counts()[0]/y_test.value_counts()[1]  # 1.919


# Exploring split 

# sizes 
X_train.shape
X_test.shape

#Response class value counts
y_train.value_counts()
y_test.value_counts()

# Statistics
X_train.describe()  
X_test.describe()


# Model selection


# Linear Regression:

from sklearn.linear_model import LinearRegression
   
LR = LinearRegression(fit_intercept=(True))

LR.fit(X_train, y_train)

#identify feature coefficients * intercept 
LR.coef_
LR.intercept_


LR.score(X_train, y_train)
LR.score(X_test, y_test)


# Linear regression used when response variable is continuous. In this case, it is 
# binary (0 or 1). Poor training score and poor test score (high bias, high variance)


# Logistic Regression:

# Visualize scatter plots per feature    
import matplotlib.pyplot as plt
for chart in data_clean.columns[:-1]:

    plt.figure(figsize = (9,9))
    plt.scatter(data_clean[chart], data_clean.Outcome)

    print(chart)


from sklearn.linear_model import LogisticRegression
LOR = LogisticRegression(penalty = "l2", max_iter=300)

LOR.fit(X_train, y_train)

LOR.score(X_train, y_train)
LOR.score(X_test, y_test)

LOR.predict_proba(X_test)
LOR.predict_log_proba(X_test)
LOR.predict(X_test)


# Coeffificients for Logistic Regression         
LOR_coef = dict( [(feat, coef) for feat, coef in zip(data.columns, LOR.coef_[0])])

print(LOR_coef["Age"])

#Bar graph visual

plt.bar(LOR_coef.keys(), LOR_coef.values())



from sklearn.metrics import confusion_matrix, classification_report

preds = LOR.predict(X_test)
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot = True)
            
            
cr  = classification_report(y_test, preds, output_dict = False, digits = 3)
print(cr)


# Decision Trees

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(random_state = 12)

DT.fit(X_train, y_train)
DT.score(X_test, y_test)

# 75% accuray first pass 

# Regularization 

maxdepth = []
for parameter in range(1,20):
    
    DT_regularized = DecisionTreeClassifier(random_state = 12,
                                        max_depth = parameter)
    DT_regularized.fit(X_train, y_train)
    maxdepth.append(f"Max Depth of {parameter}---> leads to Accuracy of {DT_regularized.score(X_test, y_test)}")
    
# Improvement of 1% when restricting trees to max_depth of 13 ===> 76.2431%

DecisionTreeClassifier(random_state = 12,
                       max_depth = 13).fit(X_train, y_train).score(X_test, y_test)

DT_regularized = DecisionTreeClassifier(random_state = 12,
                                    max_depth = 13,
                                    )

DT_regularized.fit(X_train, y_train)
DT_regularized.score(X_test, y_test)

# VIsualize the tree splits / nodes
from sklearn import tree
print(tree.export_text(DT_regularized))

plt.figure(figsize = (100,100))
print(tree.plot_tree(DT_regularized, filled = True, rounded = True, class_names=["1", "0"], feature_names=X.columns))

# Ensemble Methods

# 1 : Random Forest

from sklearn.ensemble import RandomForestClassifier


# optimal number of trees in Random Forest 
nest = []
for tr in range(3, 300):
    RF = RandomForestClassifier(random_state=(12),
                                n_estimators=tr,
                                max_depth=(13),
                                bootstrap=(True),
                                oob_score=(True)
                                )

    RF.fit(X_train, y_train)
    nest.append(f" ({RF.score(X_test, y_test)}), {tr}")

RF = RandomForestClassifier(random_state=(12),
                            n_estimators=19,
                            max_depth=(13),
                            bootstrap=(True),
                            oob_score=(True)
                            )

RF.fit(X_train, y_train)
RF.score(X_test, y_test)
RF.oob_score_
#optimal trees  = 19 --> 79.558 %
# out of bag score lower than ensemble










