import pandas as pd
import numpy as np 
data=pd.read_csv(r'C:\Users\HP\Downloads\credit\UCI_Credit_Card.csv')
data=data.drop(['ID','SEX', 'EDUCATION', 'MARRIAGE', 'AGE',],axis=1)
data=data.drop_duplicates()

X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

#KNN 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=4,p=1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 


#SVM
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 

### Using for Loop ###

# Define the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=4, p=1),
    "SVM": SVC()
}

# Initialize a list to store results
results = []

# Loop over classifiers
for name, classifier in classifiers.items():
    # Train the classifier
    classifier.fit(X_train, y_train)
    
    # Predict the test set results
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy
    ac = accuracy_score(y_test, y_pred)
    
    # Append the result to the list
    results.append({"Model": name, "Accuracy": ac})

# Convert the results list to a DataFrame
result_df = pd.DataFrame(results)

# Display the result DataFrame
print(result_df)





