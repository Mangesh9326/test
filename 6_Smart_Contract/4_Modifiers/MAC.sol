// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract multiplemodifier {

    address admin;

    struct employee {
        uint emp_id;
        string emp_name;
        uint age;
    }

    employee e;

    constructor() {
        admin = msg.sender;
    }

    modifier isAdmin() {
        require(admin == msg.sender, "Not authorized");
        _;
    }

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
    )
        public
        isAdmin
        isExperienced(2)
    {
        e.emp_id = _empid;
        e.emp_name = _empname;
        e.age = _empage;
    }
}