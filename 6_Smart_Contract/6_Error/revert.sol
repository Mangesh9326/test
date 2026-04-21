// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

contract RevertStatement {
    function checkOverflow(uint _num1, uint _num2)
    public
    pure
    returns (string memory, uint)
    {
        uint sum = _num1 + _num2;
        if (sum > 255) {
            revert("Overflow Exist");
        } else {
            return ("No Overflow", sum);
        }
    }
}