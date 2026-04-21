// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

contract f {
    uint public a;
    constructor(uint _a) {
        a = _a;
    }
    function fact() public view returns (uint256) {
        uint256 r = 1;
        for (uint256 i = 1; i <= a; i++) {
            r *= i;
        }
        return r;
    }
}