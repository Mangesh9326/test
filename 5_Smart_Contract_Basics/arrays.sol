// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

contract SimpleArray {
    uint[] public numbers;
    function addNumber(uint _number) public {
        numbers.push(_number);
    }
    function getNumbers() public view returns (uint[] memory) {
        return numbers;
    }
    function getNumberCount() public view returns (uint) {
        return numbers.length;
    }
}