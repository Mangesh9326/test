// SPDX-License-Identifier: MIT
pragma solidity ^0.8;

contract Weekday {
    function getWeekday(uint8 day) public pure returns (string memory) {
        if (day == 1) {
            return "Monday";
        } else if (day == 2) {
            return "Tuesday";
        } else if (day == 3) {
            return "Wednesday";
        } else if (day == 4) {
            return "Thursday";
        } else if (day == 5) {
            return "Friday";
        } else if (day == 6) {
            return "Saturday";
        } else if (day == 7) {
            return "Sunday";
        } else {
            return "Invalid input";
        }
    }
}