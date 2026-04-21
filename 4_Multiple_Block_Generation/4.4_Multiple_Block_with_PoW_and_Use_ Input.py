import hashlib
import json
import time

DIFFICULTY = 4


def compute_hash(block):
    block_copy = dict(block)
    block_copy.pop("hash", None)
    block_string = json.dumps(block_copy, sort_keys=True).encode()
    return hashlib.sha256(block_string).hexdigest()


def compute_merkle_root(transactions):
    if not transactions:
        return hashlib.sha256("".encode()).hexdigest()

    hashes = [
        hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest()
        for tx in transactions
    ]

    while len(hashes) > 1:
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1])

        hashes = [
            hashlib.sha256((hashes[i] + hashes[i + 1]).encode()).hexdigest()
            for i in range(0, len(hashes), 2)
        ]

    return hashes[0]


def mine_block(block):
    block["nonce"] = 0
    computed_hash = compute_hash(block)

    while not computed_hash.startswith("0" * DIFFICULTY):
        block["nonce"] += 1
        computed_hash = compute_hash(block)

    return computed_hash


def create_transaction(sender, receiver, amount):
    return {
        "sender": sender,
        "receiver": receiver,
        "amount": amount
    }


def create_block(index, transactions, previous_hash):
    block = {
        "index": index,
        "timestamp": time.time(),
        "transactions": transactions,
        "merkle_root": compute_merkle_root(transactions),
        "previous_hash": previous_hash,
        "nonce": 0
    }

    block["hash"] = mine_block(block)
    return block


def get_transactions_from_user():
    transactions = []
    print("\nEnter transactions (type 'done' to finish):")

    while True:
        sender = input("Sender (or 'done'): ")
        if sender.lower() == "done":
            break

        receiver = input("Receiver: ")
        amount = float(input("Amount: "))
        transactions.append(create_transaction(sender, receiver, amount))
        print("Transaction added.\n")

    return transactions


def generate_blockchain(num_blocks):
    blockchain = []

    # Genesis block
    genesis_block = create_block(0, [], "0")
    blockchain.append(genesis_block)

    for i in range(1, num_blocks):
        print(f"\n--- Block {i} ---")
        transactions = get_transactions_from_user()
        previous_hash = blockchain[-1]["hash"]

        block = create_block(i, transactions, previous_hash)
        blockchain.append(block)

    return blockchain


if __name__ == "__main__":
    num_blocks = int(input("Enter number of blocks to create: "))
    chain = generate_blockchain(num_blocks)

    print("\n=== BLOCKCHAIN ===")
    for block in chain:
        print(json.dumps(block, indent=4))
        print()
