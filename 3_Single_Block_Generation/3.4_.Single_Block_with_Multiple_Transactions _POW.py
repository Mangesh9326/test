import hashlib
import datetime
import json
import time
import random
import sys


class Block:
    def __init__(self, transactions, difficulty=2, previous_hash="0" * 64):
        self.index = 0  # Single block example
        self.timestamp = str(datetime.datetime.now())
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = 0
        self.difficulty = difficulty
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "transactions": self.transactions,
                "previous_hash": self.previous_hash,
                "nonce": self.nonce,
            },
            sort_keys=True
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine(self):
        print(f"Mining block with difficulty {self.difficulty}...")
        start_time = time.time()
        required_prefix = "0" * self.difficulty

        while not self.hash.startswith(required_prefix):
            self.nonce = random.randint(0, sys.maxsize)
            self.hash = self.compute_hash()

        end_time = time.time()
        print(f"Block mined: {self.hash}")
        print(f"Nonce found: {self.nonce}")
        print(f"Time taken to mine: {end_time - start_time:.2f} seconds\n")

    def __str__(self):
        return (
            f"Block index: {self.index}\n"
            f"Timestamp: {self.timestamp}\n"
            f"Transactions:\n"
            + "\n".join(f" - {tx}" for tx in self.transactions)
            + "\n"
            f"Previous hash: {self.previous_hash}\n"
            f"Nonce: {self.nonce}\n"
            f"Hash: {self.hash}\n"
        )


# Example transactions
transactions = [
    "Alice pays Bob 5 BTC",
    "Bob pays Charlie 2 BTC",
    "Charlie pays Dave 1 BTC",
]

# Create and mine block
block = Block(transactions, difficulty=4)
block.mine()

print(block)
