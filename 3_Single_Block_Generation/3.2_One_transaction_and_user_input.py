import hashlib
from datetime import datetime
import json
def create_blockchain_transaction_block():
    print("--- Blockchain Transaction Block Creation ---")
    sender = input("Enter Sender Address (Public Key): ")
    recipient = input("Enter Recipient Address (Public Key): ")
    amount_str = input("Enter Transaction Amount: ")
    try:
        amount = float(amount_str)
    except ValueError:
        print("\n[ERROR] Transaction Amount must be a valid number.")
        return None
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    transaction_block = {
        "timestamp": timestamp,
        "sender": sender,
        "recipient": recipient,
        "amount": amount,
        "previous_hash": "0" * 64,  # Placeholder for the previous block's hash
        "nonce": 0  # Placeholder for Proof-of-Work (Mining)
    }
    block_string = json.dumps(transaction_block, sort_keys=True).encode()
    transaction_block["current_hash"] = hashlib.sha256(block_string).hexdigest()
    print("\n--- Secure Transaction Block Created ---")
    for key, value in transaction_block.items():
        print(f" {key}: {value}")
    return transaction_block
# Run the function
new_block = create_blockchain_transaction_block()
