import hashlib 
import json 
import time 
 
 
def create_transaction(sender, recipient, amount): 
    return { 
        "sender": sender, 
        "recipient": recipient, 
        "amount": amount 
    } 
 
def compute_hash(block): 
    block_string = json.dumps(block, sort_keys=True).encode() 
    return hashlib.sha256(block_string).hexdigest() 
 
 
def create_block(index, transactions, previous_hash): 
    block = { 
        "index": index, 
        "timestamp": time.time(), 
        "transactions": transactions, 
        "previous_hash": previous_hash, 
    } 
 
    block["hash"] = compute_hash(block) 
    return block 
 
def generate_blockchain(num_blocks, tx_per_block=2): 
    blockchain = [] 
 
    genesis_block = create_block(0, [], "0") 
 
    blockchain.append(genesis_block) 
 
    for i in range(1, num_blocks): 
        transactions = [ 
            create_transaction(f"User{i}", f"User{i+1}", i * 10 + j) 
            for j in range(tx_per_block) 
        ] 
 
        previous_hash = blockchain[-1]["hash"] 
        new_block = create_block(i, transactions, previous_hash) 
        blockchain.append(new_block) 
 
    return blockchain 
 
 
if __name__ == "__main__": 
    chain = generate_blockchain(5, tx_per_block=3) 
 
    for block in chain: 
        print(json.dumps(block, indent=4), "\n")
