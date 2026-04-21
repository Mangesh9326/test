import hashlib
from datetime import datetime

# Global difficulty and blockchain list
DIFFICULTY_LEVEL = 0
BLOCKCHAIN = []


class Block:
    def __init__(self, data, prev_hash="0" * 64):
        self.blockId = len(BLOCKCHAIN)
        self.timestamp = datetime.now().strftime("%d-%b-%y %H:%M:%S.%f")
        self.data = data
        self.prev_hash = prev_hash
        self.nonce = 0
        self.hash = self.hashBlock()

    def hashBlock(self):
        block_string = (
            str(self.blockId)
            + str(self.data)
            + str(self.prev_hash)
            + str(self.nonce)
        )

        self.hash = hashlib.sha256(block_string.encode()).hexdigest()

        # Proof of work: increment nonce until hash has required leading zeros
        while self.hash[:DIFFICULTY_LEVEL] != "0" * DIFFICULTY_LEVEL:
            self.nonce += 1
            block_string = (
                str(self.blockId)
                + str(self.data)
                + str(self.prev_hash)
                + str(self.nonce)
            )
            self.hash = hashlib.sha256(block_string.encode()).hexdigest()

        return self.hash

    def __str__(self):
        return (
            f"\nBlock ID: {self.blockId}\n"
            f"Timestamp: {self.timestamp}\n"
            f"Data: {self.data}\n"
            f"Previous Hash: {self.prev_hash}\n"
            f"Nonce: {self.nonce}\n"
            f"Hash: {self.hash}\n"
        )


# --- Main Program ---

# Get user input
data = input("Enter the data for the block: ")
DIFFICULTY_LEVEL = int(
    input("Enter difficulty level (number of leading zeros): ")
)

# Create genesis block or next block
prev_hash = BLOCKCHAIN[-1].hash if BLOCKCHAIN else "0" * 64
block = Block(data, prev_hash)

BLOCKCHAIN.append(block)

print(block)
