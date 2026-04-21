const contractABI = [
    "function set(uint256 x) public",
    "function get() public view returns (uint256)"
];

const contractAddress = "0x04B88BE673D3E241b602589fa08754a963B9XXXX";

let provider;
let signer;
let contract;

async function init() {
    if (typeof window.ethereum === "undefined") {
        alert("MetaMask is not installed!");
        return;
    }

    await window.ethereum.request({ method: "eth_requestAccounts" });

    provider = new ethers.providers.Web3Provider(window.ethereum);
    signer = provider.getSigner();
    contract = new ethers.Contract(contractAddress, contractABI, signer);
}

async function setValue() {
    const value = document.getElementById("inputValue").value;

    if (!value) {
        alert("Enter a number first!");
        return;
    }

    try {
        const tx = await contract.set(value);
        await tx.wait();
        alert("Value set successfully!");
    } catch (err) {
        console.error(err);
        alert("Transaction failed!");
    }
}

async function getValue() {
    try {
        const value = await contract.get();
        document.getElementById("storedValue").innerText = value.toString();
    } catch (err) {
        console.error(err);
        alert("Failed to get value!");
    }
}

window.onload = init;