const EventEmitter = require('events');

class CityEventBus extends EventEmitter {
    constructor() {
        super();
        this.setMaxListeners(50); // High limit for multiple scalable agents
    }
}

const eventBus = new CityEventBus();
module.exports = eventBus;