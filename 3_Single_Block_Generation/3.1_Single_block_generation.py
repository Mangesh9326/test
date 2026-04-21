import hashlib
import time
import json

class Block:
    def __init__(self, index, data, previous_hash):
        self.index = index
        self.timestamp = time.time()
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "data": self.data,
                "previous_hash": self.previous_hash,
                "nonce": self.nonce,
            },
            sort_keys=True
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

# Create genesis block (first block, previous_hash = "0")
genesis_block = Block(0, "Genesis Block", "0")

print("Block index:", genesis_block.index)
print("Block timestamp:", genesis_block.timestamp)
print("Block data:", genesis_block.data)
print("Previous hash:", genesis_block.previous_hash)
print("Current hash:", genesis_block.hash)
