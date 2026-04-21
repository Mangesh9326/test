// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

contract Contract_A {
    uint256 value;
    constructor(uint256 n) {
        value = n;
    }

    function getValue() public view returns (uint256) {
        return value;
    }
}

contract Contract_B is Contract_A(22) {
    function incrementValue(uint256 x) public {
        value += x;
    }
}

contract Contract_C is Contract_A(25) {
    function decrementValue(uint256 x) public {
        value -= x;
    }
}