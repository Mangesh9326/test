pragma solidity >0.4.0 <0.6.0;

contract Crud {

    struct User {
        uint id;
        string name;
        string email;
        uint age;
    }

    User[] users;
    uint nextId;

    function create(
        string memory name,
        string memory email,
        uint age
    ) public {
        users.push(User(nextId, name, email, age));
        nextId++;
    }

    function read(uint id)
        public
        view
        returns (
            uint,
            string memory,
            string memory,
            uint
        )
    {
        uint i = find(id);

        return (
            users[i].id,
            users[i].name,
            users[i].email,
            users[i].age
        );
    }

    function update(
        uint id,
        string memory name,
        string memory email,
        uint age
    ) public {
        uint i = find(id);

        users[i].name = name;
        users[i].email = email;
        users[i].age = age;
    }

    function destroy(uint id) public {
        uint i = find(id);
        delete users[i];
    }

    function find(uint id) internal view returns (uint) {
        for (uint i = 0; i < users.length; i++) {
            if (users[i].id == id) {
                return i;
            }
        }

        revert("User does not exist");
    }
}