// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

library lib {
    function increment(uint val) public pure returns (uint) {
        return val + 1;
    }

    function decrement(uint val) public pure returns (uint) {
        return val - 1;
    }

    function incrementByValue(uint val, uint x) public pure returns (uint) {
        return val + x;
    }
    
    function decrementByValue(uint val, uint x) public pure returns (uint) {
        return val - x;
    }
}