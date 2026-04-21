import hashlib
import json
import time


def hash_block(block):
    block_string = json.dumps(block, sort_keys=True).encode()
    return hashlib.sha256(block_string).hexdigest()


def create_block(index, previous_hash, transactions):
    block = {
        "index": index,
        "timestamp": time.time(),
        "transactions": transactions,
        "previous_hash": previous_hash,
    }
    block["hash"] = hash_block(block)
    return block


def is_valid_block(block, previous_block):
    if block["index"] != previous_block["index"] + 1:
        print("Invalid index")
        return False

    if block["previous_hash"] != previous_block["hash"]:
        print("Previous hash mismatch")
        return False

    recalculated_hash = hash_block(
        {k: block[k] for k in block if k != "hash"}
    )

    if recalculated_hash != block["hash"]:
        print("Hash mismatch")
        return False

    return True


def is_valid_chain(chain):
    for i in range(1, len(chain)):
        if not is_valid_block(chain[i], chain[i - 1]):
            print(f"Block {i} is invalid")
            return False
    return True


chain = []
chain.append(create_block(0, "0", []))

for i in range(1, 3):
    transactions = [f"User{i} pays User{i+1} {10 * j}" for j in range(1, 4)]
    chain.append(create_block(i, chain[-1]["hash"], transactions))


for block in chain:
    print(json.dumps(block, indent=4))
    print()


print("Is blockchain valid?", is_valid_chain(chain))
