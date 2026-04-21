// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

contract modifierWithArg {
    struct employee {
        uint emp_id;
        string emp_name;
        uint age;
    }
    employee e;
    modifier isExperienced(uint exp) {
        if (exp >= 5) {
            _;
        } else {
            revert("Must have a minimum of 5 years of experience");
        }
    }
    function enterDetails(
        uint _empid,
        string memory _empname,
        uint _empage
        ) public isExperienced(2) {
            e.emp_id = _empid;
            e.emp_name = _empname;
            e.age = _empage;
        }
}