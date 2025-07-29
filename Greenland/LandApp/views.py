from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import numpy as np
import base64
import os
from datetime import datetime
import json
from web3 import Web3, HTTPProvider
import ipfsApi
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from keras.models import Sequential, load_model, Model

global username, usersList, transactionList
global contract, web3
api = ipfsApi.Client(host='http://127.0.0.1', port=5001)

#function to call contract
def getContract():
    global contract, web3
    blockchain_address = 'http://127.0.0.1:9545'
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'GreenLand.json' #GreenLand contract file
    deployed_contract_address = '0xd374Cb05bd6187D6cF905D7bBD85f2b704fBDD29' #contract address
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi)
getContract()

def getUsersList():
    global usersList, contract
    usersList = []
    count = contract.functions.getUserCount().call()
    for i in range(0, count):
        user = contract.functions.getUsername(i).call()
        password = contract.functions.getPassword(i).call()
        phone = contract.functions.getPhone(i).call()
        email = contract.functions.getEmail(i).call()
        user_type = contract.functions.getUserType(i).call()
        usersList.append([user, password, phone, email, user_type])

def getTransactionList():
    global transactionList, contract
    transactionList = []
    count = contract.functions.getTransactionCount()().call()
    for i in range(0, count):
        buyer_seller = contract.functions.getBuyerSeller(i).call()
        transaction_hash = contract.functions.getTransactionHash(i).call()
        upload_date = contract.functions.getDate(i).call()
        transactionList.append([buyer_seller, transaction_hash, upload_date])

getUsersList()
getTransactionList()

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


def ViewRequestStatus(request):
    if request.method == 'GET':
        global username, transactionList
        output = '<table border=1 align=center width=100%><tr><th><font size="3" color="black">Buyer Name</th><th><font size="3" color="black">IPFS Transaction Hashcode</th>'
        output += '<th><font size="3" color="black">Request Date</th>'
        output+='<th><font size="3" color="black">AI Detected Transaction Status</th><th><font size="3" color="black">Seller Decision</th></tr>'
        for i in range(len(transactionList)):
            tl = transactionList[i]
            status = tl[2].split(",")
            if tl[0] == username and len(status) == 3:
                output += '<tr><td><font size="" color="black">'+str(tl[0])+'</td><td><font size="" color="black">'+tl[1]+'</td>'
                output+='<td><font size="3" color="black">'+status[0]+'</td>'
                if status[1] == 'Fraud':
                    output += '<td><font size="3" color="red">'+status[1]+'</td>'
                else:
                    output += '<td><font size="3" color="green">'+status[1]+'</td>'
                output += '<td><font size="3" color="black">'+status[2]+'</td></tr>' 
        output += "</table><br/><br/><br/><br/>"    
        context= {'data':output}
        return render(request, 'BuyerScreen.html', context)

def AcceptRejectAction(request):
    if request.method == 'GET':
        global transactionList
        index = request.GET.get('t1', False)
        status = request.GET.get('t2', False)
        tl = transactionList[int(index)]
        tl[2] = tl[2]+","+status
        contract.functions.updateStatus(int(index), tl[2]).transact()
        context= {'data':"Buyer request successfully <font size=3 color=blue>"+status+"</font>"}
        return render(request, 'SellerScreen.html', context)

def AcceptReject(request):
    if request.method == 'GET':
        global username, transactionList
        output = '<table border=1 align=center width=100%><tr><th><font size="3" color="black">Buyer Name</th><th><font size="3" color="black">IPFS Transaction Hashcode</th>'
        output += '<th><font size="3" color="black">Request Date</th>'
        output+='<th><font size="3" color="black">AI Detected Transaction Status</th><th><font size="3" color="black">Accept Request</th><th><font size="3" color="black">Reject Request</th></tr>'
        for i in range(len(transactionList)):
            tl = transactionList[i]
            status = tl[2].split(",")
            if "Accepted" not in tl[2] and "Rejected" not in tl[2]:
                output += '<tr><td><font size="" color="black">'+str(tl[0])+'</td><td><font size="" color="black">'+tl[1]+'</td>'
                output+='<td><font size="3" color="black">'+status[0]+'</td>'
                if status[1] == 'Fraud':
                    output += '<td><font size="3" color="red">'+status[1]+'</td>'
                else:
                    output += '<td><font size="3" color="green">'+status[1]+'</td>'
                output +='<td><a href=\'AcceptRejectAction?t1='+str(i)+'&t2=Accepted\'><font size=3 color=green>Accept Request</font></a></td>'
                output +='<td><a href=\'AcceptRejectAction?t1='+str(i)+'&t2=Rejected\'><font size=3 color=red>Reject Request</font></a></td></tr>' 
        output += "</table><br/><br/><br/><br/>"    
        context= {'data':output}
        return render(request, 'SellerScreen.html', context)

def RequestLandAction(request):
    if request.method == 'POST':
        global username
        global transactionList, label_encoder, scaler, selected, lg_cls, dataset
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        hashcode = api.add_pyobj(myfile)
        upload_time = str(datetime.now())
        if os.path.exists("LandApp/static/files/"+fname):
            os.remove("LandApp/static/files/"+fname)        
        with open("LandApp/static/files/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        testData = pd.read_csv("LandApp/static/files/"+fname)
        for j in range(len(label_encoder)):
            le = label_encoder[j]
            testData[le[0]] = pd.Series(le[1].transform(testData[le[0]].astype(str)))#encode all str columns to numeric        
        testData.fillna(dataset.mean(), inplace = True)#replace missing values using mean imputation
        testData.drop(['Index'], axis = 1,inplace=True)
        testData = testData[selected]
        testData = testData.values
        testData = scaler.transform(testData)
        #testData = np.reshape(testData, (testData.shape[0], testData.shape[1], 1, 1))
        cnn_model = load_model("model/cnn_weights.hdf5")
        predict = lg_cls.predict(testData)
        status = "Non-Fraud"
        if predict == 1:
            status = "Fraud"
        msg = contract.functions.saveTransaction(username, hashcode, upload_time+","+status).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)
        transactionList.append([username, hashcode, upload_time+","+status])        
        status = '<font size="3" color="blue">Buyer Request IPFS File Saving Hashcode = '+str(hashcode)+"</font><br/>"
        status += str(tx_receipt)
        context= {'data':status}
        return render(request, 'RequestLand.html', context)

def RequestLand(request):
    if request.method == 'GET':
        return render(request, 'RequestLand.html', {})

def SellerLogin(request):
    if request.method == 'GET':
        return render(request, 'SellerLogin.html', {})

def BuyerLogin(request):
    if request.method == 'GET':
        return render(request, 'BuyerLogin.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def RegisterAction(request):
    if request.method == 'POST':
        global usersList
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        user_type = request.POST.get('t5', False)
        count = contract.functions.getUserCount().call()
        status = "none"
        for i in range(0, count):
            user1 = contract.functions.getUsername(i).call()
            if username == user1:
                status = "exists"
                break
        if status == "none":
            msg = contract.functions.saveUser(username, password, contact, email, user_type).transact()
            tx_receipt = web3.eth.waitForTransactionReceipt(msg)
            usersList.append([username, password, contact, email, user_type])
            context= {'data':'<font size="3" color="blue">Signup Process Completed</font><br/>'+str(tx_receipt)}       
            return render(request, 'Register.html', context)
        else:
            context= {'data':'Given username already exists'}
            return render(request, 'Register.html', context)

def BuyerLoginAction(request):
    if request.method == 'POST':
        global username, contract, usersList
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        status = 'none'
        for i in range(len(usersList)):
            ulist = usersList[i]
            user1 = ulist[0]
            pass1 = ulist[1]            
            if user1 == username and pass1 == password and ulist[4] == "Buyer":
                status = "success"
                break
        if status == 'success':
            output = 'Welcome '+username
            context= {'data':output}
            return render(request, "BuyerScreen.html", context)
        if status == 'none':
            context= {'data':'Invalid login details'}
            return render(request, 'BuyerLogin.html', context)

def SellerLoginAction(request):
    if request.method == 'POST':
        global username, contract, usersList
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        status = 'none'
        for i in range(len(usersList)):
            ulist = usersList[i]
            user1 = ulist[0]
            pass1 = ulist[1]            
            if user1 == username and pass1 == password and ulist[4] == "Seller":
                status = "success"
                break
        if status == 'success':
            output = 'Welcome '+username
            context= {'data':output}
            return render(request, "SellerScreen.html", context)
        if status == 'none':
            context= {'data':'Invalid login details'}
            return render(request, 'SellerLogin.html', context)
