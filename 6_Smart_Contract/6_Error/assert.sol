// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

contract AssertStatement {
    bool result;
    function checkOverflow(uint _num1, uint _num2) public {
        uint sum = _num1 + _num2;
        assert(sum <= 255);
        result = true;
    }
    
    function getResult() public view returns (string memory) {
        if (result == true) {
            return "No Overflow";
        } else {
            return "Overflow exists";
        }
    }
}