// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

contract mappingExample {
    mapping(int => string) details;
    function addDetails(int id, string memory name) public {
        details[id] = name;
    }
    
    function updateDetails(int id, string memory name) public {
        details[id] = name;
    }

    function getDetails(int id) public view returns (string memory) {
        return details[id];
    }
}