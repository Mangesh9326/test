import { io } from 'socket.io-client';
import { useGlobalStore } from '../store/useGlobalStore';
import { useTrafficStore } from '../store/useTrafficStore';

class SocketService {
  constructor() {
    this.socket = null;
  }

  connect() {
    if (this.socket) return;

    // Connect to the backend Node.js server
    this.socket = io('http://localhost:5000');

    this.socket.on('connect', () => {
      console.log(`[SOCKET] Connected to backend ID: ${this.socket.id}`);
    });

    // Listen for Global System Ticks from Replay Engine
    this.socket.on('SYSTEM_TICK', (data) => {
      useGlobalStore.getState().setSimulationTime(`00:${data.timestamp < 10 ? '0' : ''}${data.timestamp}`);
    });

    // Listen for Multi-Agent Decision Plan broadcasts
    this.socket.on('DECISION_COMMAND', (plan) => {
      console.log('[SOCKET] Received Decision Intelligence Plan:', plan);
      useGlobalStore.getState().addAlert({
        id: plan.eventId,
        type: plan.priority,
        message: plan.commandCenterDirective
      });
      useGlobalStore.getState().addRecommendation(plan.commandCenterDirective);
    });

    // Listen for isolated Traffic updates
    this.socket.on('TRAFFIC_UPDATE', (trafficData) => {
      useTrafficStore.getState().updateTrafficData(trafficData);
    });

    this.socket.on('disconnect', () => {
      console.log('[SOCKET] Disconnected from backend.');
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export default new SocketService();