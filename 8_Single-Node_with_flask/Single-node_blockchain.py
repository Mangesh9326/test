import datetime
import hashlib
import json
import requests

from flask import Flask, jsonify, request
from uuid import uuid4
from urllib.parse import urlparse


class Blockchain:

    def __init__(self):
        self.chain = []
        self.transactions = []
        self.nodes = set()

        self.create_block(proof=1, previous_hash="0")

    def create_block(self, proof, previous_hash):
        block = {
            "index": len(self.chain) + 1,
            "timestamp": str(datetime.datetime.now()),
            "proof": proof,
            "previous_hash": previous_hash,
            "transactions": self.transactions,
        }

        self.transactions = []
        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False

        while not check_proof:
            hash_value = hashlib.sha256(
                str(new_proof**2 - previous_proof**2).encode()
            ).hexdigest()

            if hash_value[:5] == "00000":
                check_proof = True
            else:
                new_proof += 1

        return new_proof

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1

        while block_index < len(chain):
            block = chain[block_index]

            if block["previous_hash"] != self.hash(previous_block):
                return False

            previous_proof = previous_block["proof"]
            proof = block["proof"]

            hash_value = hashlib.sha256(
                str(proof**2 - previous_proof**2).encode()
            ).hexdigest()

            if hash_value[:5] != "00000":
                return False

            previous_block = block
            block_index += 1

        return True

    def add_transaction(self, sender, receiver, amount):
        self.transactions.append(
            {"sender": sender, "receiver": receiver, "amount": amount}
        )

        previous_block = self.get_previous_block()
        return previous_block["index"] + 1

    def add_node(self, address):
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)

    def replace_chain(self):
        network = self.nodes
        longest_chain = None
        max_length = len(self.chain)

        for node in network:
            response = requests.get(f"http://{node}/get_chain")

            if response.status_code == 200:
                length = response.json()["length"]
                chain = response.json()["chain"]

                if length > max_length and self.is_chain_valid(chain):
                    max_length = length
                    longest_chain = chain

        if longest_chain:
            self.chain = longest_chain
            return True

        return False


# ---------------- FLASK APP ---------------- #

app = Flask(__name__)
blockchain = Blockchain()

node_address = str(uuid4()).replace("-", "")


@app.route("/mine_block", methods=["GET"])
def mine_block():
    previous_block = blockchain.get_previous_block()
    previous_proof = previous_block["proof"]

    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(previous_block)

    block = blockchain.create_block(proof, previous_hash)

    response = {
        "message": "Block successfully mined",
        "index": block["index"],
        "timestamp": block["timestamp"],
        "proof": block["proof"],
        "previous_hash": block["previous_hash"],
        "transactions": block["transactions"],
    }

    return jsonify(response), 200


@app.route("/get_chain", methods=["GET"])
def get_chain():
    response = {
        "chain": blockchain.chain,
        "length": len(blockchain.chain),
    }
    return jsonify(response), 200


@app.route("/is_valid", methods=["GET"])
def is_valid():
    is_valid_chain = blockchain.is_chain_valid(blockchain.chain)

    if is_valid_chain:
        response = {"message": "Blockchain is valid"}
    else:
        response = {"message": "Blockchain is NOT valid"}

    return jsonify(response), 200


@app.route("/add_transaction", methods=["POST"])
def add_transaction():
    json_data = request.get_json()

    transaction_keys = ["sender", "receiver", "amount"]

    if not all(key in json_data for key in transaction_keys):
        return "Missing transaction fields", 400

    index = blockchain.add_transaction(
        json_data["sender"],
        json_data["receiver"],
        json_data["amount"],
    )

    response = {
        "message": f"Transaction will be added to Block {index}"
    }

    return jsonify(response), 201


@app.route("/connect_nodes", methods=["POST"])
def connect_nodes():
    json_data = request.get_json()
    nodes = json_data.get("nodes")

    if nodes is None:
        return "No nodes provided", 400

    for node in nodes:
        blockchain.add_node(node)

    response = {
        "message": "All nodes are now connected",
        "total_nodes": list(blockchain.nodes),
    }

    return jsonify(response), 201


@app.route("/replace_chain", methods=["GET"])
def replace_chain():
    is_replaced = blockchain.replace_chain()

    if is_replaced:
        response = {
            "message": "Chain was replaced",
            "new_chain": blockchain.chain,
        }
    else:
        response = {
            "message": "Chain is already the longest",
            "chain": blockchain.chain,
        }

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
