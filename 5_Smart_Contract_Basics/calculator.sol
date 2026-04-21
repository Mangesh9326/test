// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

contract n {
    uint public na;
    uint public nb;
    constructor(uint _na, uint _nb) {
        na = _na;
        nb = _nb;
    }
    
    function add() public view returns (uint) {
        return na + nb;
    }
    function sub() public view returns (uint) {
        require(na >= nb, "Subtraction results in negative value");
        return na - nb;
    }
    function mul() public view returns (uint) {
        return na * nb;
    }

    function div() public view returns (uint) {
        require(nb != 0, "Division by zero is not allowed");
        return na / nb;
    }
    function getAllResults() public view returns (uint, uint, uint, uint) {
        uint sum = na + nb;
        uint diff = na >= nb ? na - nb : 0;
        uint prod = na * nb;
        uint quotient = nb != 0 ? na / nb : 0;
        return (sum, diff, prod, quotient);
    }
    function setValues(uint _na, uint _nb) public {
        na = _na;
        nb = _nb;
    }
}