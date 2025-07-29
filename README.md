# 🌱 GREENLAND: A Secure Land Registration Scheme

This project integrates **Blockchain**, **IPFS**, and **AI** technologies to provide a secure and decentralized land registration system for the agriculture industry.

---

## ⚙️ Prerequisites

- [Truffle](https://trufflesuite.com/)
- [IPFS](https://docs.ipfs.io/)
- Python 3.x and Django
- `ipfsapi` Python module (or equivalent)

---

## 🚀 How to Run the Project

Follow the steps in order:

---

### 1️⃣ Run the Blockchain

#### 1.1 Open Truffle Development Console
truffle develop
1.2 Inside the Truffle Console, run:migrate
This compiles and deploys smart contracts to the local blockchain.

### 2️⃣ Start the IPFS API
2.1 Initialize IPFS (only once)ipfs init
2.2 Start the IPFS Daemon :ipfs daemon
If you're using a Python wrapper or custom IPFS server:python ipfs_api_server.py

### 3️⃣ Run the Django Server
Make sure you're in the Django project folder. Then run:python manage.py runserver
Starting development server at http://127.0.0.1:8000/

###4️⃣ Open in Browser
Click or open the URL in browser:http://127.0.0.1:8000/index.html
