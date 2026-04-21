// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract RequireStatement {

    // Validates input is uint8 (0 to 255)
    function checkInput(uint _input) public pure returns (string memory) {
        require(_input >= 0, "invalid uint8");
        require(_input <= 255, "invalid uint8");
        return "Input is Uint8";
    }

    // Validates input is odd
    function odd(uint _input) public pure returns (bool) {
        require(_input % 2 != 0, "Input is not odd");
        return true;
    }
}