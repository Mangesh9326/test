// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

contract PrimeCheck {
    uint public a;
    constructor(uint _a) {
        a = _a;
    }
    function isPrime() public view returns (bool) {
        if (a <= 1) {
            return false;
        }
        for (uint256 i = 2; i * i <= a; i++) {
            if (a % i == 0) {
                return false;
            }
        }
        return true;
    }
}
