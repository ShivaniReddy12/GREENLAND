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
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import os

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
dataset.fillna(dataset.mean(), inplace = True)#replace missing values

Y = dataset['FLAG'].ravel()
dataset.drop(['Unnamed: 0', 'Index', 'FLAG'], axis = 1,inplace=True)
columns = dataset.columns
X = dataset.values

selector = SelectKBest(f_classif, k="all") # k is the number of features to be selected
X = selector.fit_transform(X, Y)
scores = selector.scores_

feature_scores = list(zip(scores,columns))
sorted_feature_scores = sorted(feature_scores,reverse=True)
selected = []
for i in range(len(sorted_feature_scores)):
    arr = sorted_feature_scores[i]
    if arr[0] > 0:
        selected.append([arr[0], arr[1]])
selected = sorted(selected,reverse=True)
selected = np.asarray(selected)
selected = selected[:,1]
selected = selected[0:17]
print(dataset.shape)

dataset = dataset[selected]
print(dataset.shape)
X = dataset.values
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(np.unique(Y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
#data = np.asarray([X_train, X_test, y_train, y_test])
#np.save("model/data", data)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data


lr_cls = LogisticRegression(max_iter=300, penalty="l1", solver="saga")
#training on train data
lr_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = lr_cls.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

svm_cls = svm.SVC(C=3.0, kernel="sigmoid")
#training on train data
svm_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = svm_cls.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

#training Random Forest ML algorithm on 80% training data and then evaluating performance on 20% test data
rf_cls = RandomForestClassifier(n_estimators=100, max_features="sqrt", max_depth=5)
#training on train data
rf_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = rf_cls.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

#training XGBClassifier ML algorithm on 80% training data and then evaluating performance on 20% test data
xg_cls = XGBClassifier(max_depth=4, scale_pos_weight=1)
#training on train data
xg_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = xg_cls.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

#training LGBMClassifier ML algorithm on 80% training data and then evaluating performance on 20% test data
lg_cls = LGBMClassifier(learning_rate=0.1)
#training on train data
lg_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = lg_cls.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

cnn_model = Sequential()
cnn_model.add(Convolution2D(32, (1, 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train1, y_train1, batch_size = 16, epochs = 30, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")

predict = cnn_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test1, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)














