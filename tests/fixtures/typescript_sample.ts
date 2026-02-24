import { readFile } from "fs/promises";

class BaseModel {
    constructor(id) {
        this.id = id;
    }

    validate() {
        return this.id > 0;
    }
}

class User extends BaseModel {
    constructor(id, name) {
        super(id);
        this.name = name;
    }

    toJSON() {
        return JSON.stringify({ id: this.id, name: this.name });
    }

    log() {
        console.log(this.toJSON());
    }
}

function createUser(id, name) {
    const user = new User(id, name);
    user.validate();
    return user;
}

const loadUser = async (path) => {
    const data = await readFile(path, "utf-8");
    const parsed = JSON.parse(data);
    return createUser(parsed.id, parsed.name);
};
