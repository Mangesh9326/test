// SPDX-License-Identifier: MIT
ragma solidity ^0.8;

contract StructExample {
    struct Employee {
        uint emp_id;
        string emp_name;
        uint age;
    }

    uint[] public dynamicArray;

    function add(uint value) public {
        dynamicArray.push(value);
    }

    function getArray() public view returns (uint[] memory) {
        return dynamicArray;
    }

    function staticArrayExample() public pure returns (uint) {
        uint[3] memory c = [uint(1), 2, 3]; 
        assert(c.length == 3);
        return c.length;
    }
}