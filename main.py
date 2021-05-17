import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier

training_data = pd.read_csv("train.csv")
#testing_data = pd.read_csv("test.csv")

def get_nulls(training):
    print("Training Data:")
    print(pd.isnull(training).sum(), "\n")

get_nulls(training_data)

# Drop the cabin column, as there are too many missing values
# Drop the ticket numbers too, as there are too many categories
# Drop names as they won't really help predict survivors

training_data.drop(labels=['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

# Taking the mean/average value would be impacted by the skew
# so we should use the median value to impute missing values

training_data["Age"].fillna(training_data["Age"].median(), inplace=True)
training_data["Embarked"].fillna("S", inplace=True)

print(training_data)
get_nulls(training_data)

encoder_1 = LabelEncoder()
# encode the non-numerical data, column Sex is non-numerical
encoder_1.fit(training_data["Sex"])

# Transform and replace training data
training_sex_encoded = encoder_1.transform(training_data["Sex"])
training_data["Sex"] = training_sex_encoded

encoder_2 = LabelEncoder()
# column Embarked is non-numerical
encoder_2.fit(training_data["Embarked"])

training_embarked_encoded = encoder_2.transform(training_data["Embarked"])
training_data["Embarked"] = training_embarked_encoded

# Any value we want to reshape needs be turned into array first
ages_train = np.array(training_data["Age"]).reshape(-1, 1)
fares_train = np.array(training_data["Fare"]).reshape(-1, 1)

# Scale the data using the StandardScaler, so there aren't huge fluctuations in values
# Scaler takes arrays
scaler = StandardScaler()

training_data["Age"] = scaler.fit_transform(ages_train)
training_data["Fare"] = scaler.fit_transform(fares_train)

# Now to select our training data
# PassengerId & Survived are dropped
X_features = training_data.drop(labels=['PassengerId', 'Survived'], axis=1)
# Column Survived is set as training_data for y
y_labels = training_data['Survived']
#print("y_labels:\n", y_labels)
print("X_features.head(5):\n", X_features.head(5))

# Make the train/test data from validation

X_train, X_val, y_train, y_val = train_test_split(X_features, y_labels, test_size=0.1, random_state=27)
#print("X_train:\n", X_train)

LogReg_clf = LogisticRegression()
DTree_clf = DecisionTreeClassifier()
SVC_clf = SVC()

LogReg_clf.fit(X_train, y_train)
DTree_clf.fit(X_train, y_train)
SVC_clf.fit(X_train, y_train)

LogReg_pred = LogReg_clf.predict(X_val)
DTree_pred = DTree_clf.predict(X_val)
SVC_pred = SVC_clf.predict(X_val)

# Averaging predictions
# We simply add the different predicted values of our chosen classifiers together and then divide by the total number of classifiers,
# using floor division to get a whole value.
averaged_preds = (LogReg_pred + DTree_pred + SVC_pred)//3
acc = accuracy_score(y_val, averaged_preds)
print("accuracy_score = ", acc)

# hard voting method uses the predicted labels and a majority rules system
# soft voting method predicts a label based on the argmax/largest predicted value of the sum of the predicted probabilities
voting_clf = VotingClassifier(estimators=[('SVC', SVC_clf), ('DTree', DTree_clf), ('LogReg', LogReg_clf)], voting='hard')
voting_clf.fit(X_train, y_train)
preds = voting_clf.predict(X_val)
acc = accuracy_score(y_val, preds)
l_loss = log_loss(y_val, preds)
f1 = f1_score(y_val, preds)

print("Accuracy is: " + str(acc))
print("Log Loss is: " + str(l_loss))
print("F1 Score is: " + str(f1))