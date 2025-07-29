import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from lightgbm import LGBMClassifier

dataset = pd.read_csv("Dataset/transaction_dataset.csv")
#applying dataset processing technique to convert non-numeric data to numeric data
label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for j in range(len(types)):
    name = types[j]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[j], le])
dataset.fillna(dataset.mean(), inplace = True)#replace missing values using mean imputation
selected = np.load("model/selected.npy")
Y = dataset['FLAG'].ravel()
dataset.drop(['Unnamed: 0', 'Index', 'FLAG'], axis = 1,inplace=True)
columns = dataset.columns
dataset = dataset[selected]
X = dataset.values
scaler = StandardScaler()
X = scaler.fit_transform(X)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data
#training LGBMClassifier ML algorithm on 80% training data and then evaluating performance on 20% test data
lg_cls = LGBMClassifier(learning_rate=0.1)
#training on train data
lg_cls.fit(X_train, y_train)

testData = pd.read_csv("testTransactionData/2.txt")
for j in range(len(label_encoder)):
    le = label_encoder[j]
    print(le[0])
    testData[le[0]] = pd.Series(le[1].transform(testData[le[0]].astype(str)))#encode all str columns to numeric        
testData.fillna(dataset.mean(), inplace = True)#replace missing values using mean imputation
testData.drop(['Index'], axis = 1,inplace=True)
testData = testData[selected]
testData = testData.values
testData = scaler.transform(testData)
predict = lg_cls.predict(testData)[0]
status = "Non-Fraud"
if predict == 1:
    status = Fraud
print(status)    
