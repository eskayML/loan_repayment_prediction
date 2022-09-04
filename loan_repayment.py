import pandas as pd
from sklearn import ensemble as ens
from sklearn import metrics as mtr
from sklearn.model_selection import train_test_split


loan_data = pd.read_csv('loan_data.csv')
print ('DATASET HEAD')
print (loan_data.head())
print(f"SHAPE: { loan_data.shape }\n")




print ('MASKED COLUMN NAMES')
cols = loan_data.columns.values
print (cols)

print('TARGET DISTRIBUTION.')
print (loan_data[cols[-1]].value_counts())

y = loan_data.pop(cols[-1])#target
X = loan_data#features

y = y=="yes" # converting to bool, the string targets

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=0)

print ('SPLITTING DATA INTO TRAINING AND TESTING WITH A 80/20 SPLIT')

print ('Training Size:', X_train.shape,y_train.shape)
print ('Test Size:', X_test.shape,y_test.shape)

print ('FITTING THE MODEL')
print ('='*30,'\n')


rf = ens.RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)
test_pred = rf.predict(X_test)
print ('Test Accuracy: ')
print (mtr.accuracy_score(y_test,test_pred))
print ()
print ('Confusion Matrix:')
print (mtr.confusion_matrix(y_test, test_pred))
print ()
print ('Classification Report:')
print (mtr.classification_report(y_test, test_pred))