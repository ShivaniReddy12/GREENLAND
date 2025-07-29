pragma solidity >= 0.8.11 <= 0.8.11;
pragma experimental ABIEncoderV2;
//GreenLand solidity code
contract GreenLand {

    uint public userCount = 0; 
    mapping(uint => user) public userList; 
     struct user
     {
       string username;
       string password;
       string phone;
       string email;
       string buyer_seller_user_type;
     }
 
   // events 
   event userCreated(uint indexed _userId);
   
   //function  to save user details to Blockchain
   function saveUser(string memory uname, string memory pass, string memory phone, string memory em, string memory user_type) public {
      userList[userCount] = user(uname, pass, phone, em, user_type);
      emit userCreated(userCount);
      userCount++;
    }

     //get user count
    function getUserCount()  public view returns (uint) {
          return  userCount;
    }

    uint public transactionCount = 0; 
    mapping(uint => transaction) public transactionList; 
     struct transaction
     {
       string username;
       string transaction_hash;       
       string upload_date;
     }
 
   // events 
   event transactionCreated(uint indexed _transactionId);
   
   //function  to save transaction hash details to Blockchain
   function saveTransaction(string memory uname, string memory thash, string memory ud) public {
      transactionList[transactionCount] = transaction(uname, thash, ud);
      emit transactionCreated(transactionCount);
      transactionCount++;
    }

    function updateStatus(uint i, string memory status) public {
       transactionList[i].upload_date= status;
    }


    //get transaction count
    function getTransactionCount()  public view returns (uint) {
          return transactionCount;
    }

    function getUsername(uint i) public view returns (string memory) {
        user memory doc = userList[i];
	return doc.username;
    }

    function getPassword(uint i) public view returns (string memory) {
        user memory doc = userList[i];
	return doc.password;
    }

    function getPhone(uint i) public view returns (string memory) {
        user memory doc = userList[i];
	return doc.phone;
    }    

    function getEmail(uint i) public view returns (string memory) {
        user memory doc = userList[i];
	return doc.email;
    }

    function getUserType(uint i) public view returns (string memory) {
        user memory doc = userList[i];
	return doc.buyer_seller_user_type;
    }

    function getBuyerSeller(uint i) public view returns (string memory) {
        transaction memory doc =  transactionList[i];
	return doc.username;
    }

   function getTransactionHash(uint i) public view returns (string memory) {
        transaction memory doc =  transactionList[i];
	return doc.transaction_hash;
    }   
    
    function getDate(uint i) public view returns (string memory) {
        transaction memory doc =  transactionList[i];
	return doc.upload_date;
    } 
     
}