// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

import "Practical 6/Library/librarydefn.sol";

contract useLibrary {
    using lib for uint;
    
    function testIncrement(uint userVal) public pure returns (uint) {
        return lib.increment(userVal);
    }

    function testDecrement(uint userVal) public pure returns (uint) {
        return lib.decrement(userVal);
    }
    
    function testIncrementByValue(
        uint userVal,
        uint x
    ) public pure returns (uint) {
        return lib.incrementByValue(userVal, x);
    }

    function testDecrementByValue(
        uint userVal,
        uint x
        ) public pure returns (uint) {
            return lib.decrementByValue(userVal, x);
        }
}