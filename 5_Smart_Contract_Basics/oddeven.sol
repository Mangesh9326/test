// SPDX-License-Identifier: MIT
pragma solidity ^0.8;
contract EvenOddCheck {
    uint public a;
    constructor(uint _a) {
        a = _a;
    }
    function isEven() public view returns (string memory) {
        if (a % 2 == 0) {
            return "Number is even";
        } else {
            return "Number is odd";
        }
    }
}