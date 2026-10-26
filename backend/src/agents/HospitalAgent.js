const BaseAgent = require('./BaseAgent');
const pool = require('../config/db');

class HospitalAgent extends BaseAgent {
  constructor() {
    // FIX: Provide name, department string, and a valid resource object to satisfy BaseAgent's JSON.parse
    super('HospitalAgent', 'hospital', {
      ambulances: 25,
      medicalTeams: 40,
      icuBeds: 12
    });

    // Deterministic Operational Thresholds (Not medical standards)
    this.thresholds = {
      icu: { moderate: 0.70, high: 0.85, critical: 0.92 },
      emergency: { moderate: 0.60, high: 0.75, critical: 0.90 },
      queue: { moderate: 1, high: 2, critical: 3 }
    };
  }

  /**
   * Called dynamically when GET /api/hospital/agent/overview is requested.
   * No background loops. Request-driven evaluation.
   */
  async getNetworkAssessment() {
    try {
      const query = `
        SELECT h.hospital_id, h.name, h.area, h.emergency_available, h.status,
               m.total_beds, m.occupied_beds, m.available_beds,
               m.icu_total, m.icu_occupied, m.icu_available,
               m.emergency_capacity, m.emergency_occupancy, 
               m.ambulance_queue, m.average_wait_minutes, 
               m.patient_inflow, m.predicted_inflow
        FROM hospitals h
        LEFT JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id
        WHERE h.status = 'Active'
      `;
      const result = await pool.query(query);
      const hospitals = result.rows;

      let attentionCount = 0;
      let activeAssessments = 0;
      const decisions = [];

      for (const data of hospitals) {
        if (!data.total_beds) continue; // Skip if no monitoring data

        const assessment = this.evaluateHospital(data);
        if (assessment) {
          activeAssessments++;
          if (assessment.current_condition === 'HIGH' || assessment.current_condition === 'CRITICAL') {
            attentionCount++;
          }
          const savedDecision = await this.saveDecision(assessment);
          if (savedDecision) decisions.push(savedDecision);
        }
      }

      // Network Level Evaluation
      let networkCondition = 'NORMAL';
      if (attentionCount > hospitals.length * 0.3) networkCondition = 'CRITICAL';
      else if (attentionCount > 0) networkCondition = 'HIGH';

      return {
        status: 'ACTIVE',
        mode: 'REQUEST_DRIVEN',
        monitoredHospitals: hospitals.length,
        activeAssessments,
        attentionRequired: attentionCount,
        networkCondition,
        latestDecisions: decisions
      };
    } catch (error) {
      console.error('[HOSPITAL AGENT] Network assessment failed:', error);
      throw error;
    }
  }

  evaluateHospital(data) {
    const icuOccupancy = data.icu_total > 0 ? data.icu_occupied / data.icu_total : 0;
    const erOccupancy = data.emergency_capacity > 0 ? data.emergency_occupancy / data.emergency_capacity : 0;

    let condition = 'NORMAL';
    if (icuOccupancy >= this.thresholds.icu.critical || data.ambulance_queue >= this.thresholds.queue.critical) condition = 'CRITICAL';
    else if (icuOccupancy >= this.thresholds.icu.high || erOccupancy >= this.thresholds.emergency.high) condition = 'HIGH';
    else if (icuOccupancy >= this.thresholds.icu.moderate || erOccupancy >= this.thresholds.emergency.moderate) condition = 'MODERATE';

    const prediction = this.predictEmergencyLoad(data, erOccupancy);
    const { decisionType, decision, recommendation, expectedImpact } = this.generateDecision(data, condition, prediction, icuOccupancy, erOccupancy);
    const confidence = this.calculateConfidence(data, condition, prediction);

    return {
      hospital_id: data.hospital_id,
      hospital_name: data.name,
      area: data.area,
      current_condition: condition,
      emergency_level: erOccupancy >= this.thresholds.emergency.critical ? 'CRITICAL' : erOccupancy >= this.thresholds.emergency.high ? 'HIGH' : 'NORMAL',
      capacity_level: icuOccupancy >= this.thresholds.icu.critical ? 'CRITICAL' : icuOccupancy >= this.thresholds.icu.high ? 'HIGH' : 'NORMAL',
      prediction,
      decision_type: decisionType,
      decision,
      recommendation,
      expected_impact: expectedImpact,
      reason: `ER Occupancy: ${(erOccupancy * 100).toFixed(1)}%, ICU: ${(icuOccupancy * 100).toFixed(1)}%, Queue: ${data.ambulance_queue}`,
      confidence,
      status: 'ACTIVE',
      generated_at: new Date().toISOString()
    };
  }

  predictEmergencyLoad(data, erOccupancy) {
    let level = 'NORMAL';
    let horizon = 60;

    if (data.patient_inflow === 'HIGH' || data.ambulance_queue > 0) {
      if (erOccupancy > this.thresholds.emergency.high) {
        level = 'CRITICAL';
        horizon = 15;
      } else {
        level = 'HIGH';
        horizon = 30;
      }
    } else if (erOccupancy > this.thresholds.emergency.moderate) {
      level = 'MODERATE';
      horizon = 60;
    }

    return { level, horizon_minutes: horizon, confidence: level === 'CRITICAL' ? 94 : 88 };
  }

  generateDecision(data, condition, prediction, icuOccupancy, erOccupancy) {
    if (icuOccupancy >= this.thresholds.icu.critical || data.icu_available <= 2) {
      return {
        decisionType: 'ICU_CAPACITY_ALERT',
        decision: 'Prepare alternate critical-care capacity.',
        recommendation: 'Review suitable alternate receiving capacity for future non-critical incoming demand while preserving appropriate critical-care receiving capability.',
        expectedImpact: 'Reduce risk of critical-care saturation and preserve remaining ICU beds.'
      };
    } else if (data.ambulance_queue >= this.thresholds.queue.high || erOccupancy >= this.thresholds.emergency.critical) {
      return {
        decisionType: 'EMERGENCY_LOAD_RESPONSE',
        decision: 'Recommend emergency load balancing review.',
        recommendation: 'Request a review of suitable secondary receiving capacity and congestion-aware route recommendations for non-critical inbound units.',
        expectedImpact: 'Reduce projected emergency overload and stabilize queue duration.'
      };
    } else if (condition === 'HIGH' || prediction.level === 'CRITICAL') {
      return {
        decisionType: 'CAPACITY_PREPARATION',
        decision: 'Accelerate inpatient discharge protocols.',
        recommendation: 'Review pending discharges to accelerate bed turnover rates ahead of predicted peak intake.',
        expectedImpact: 'Stabilize capacity imbalance and prevent triage bottlenecks.'
      };
    }

    return {
      decisionType: 'CAPACITY_MONITORING',
      decision: 'Maintain standard operational readiness.',
      recommendation: 'Continue monitoring current hospital capacity. No immediate diversion required.',
      expectedImpact: 'Maintain regional emergency readiness.'
    };
  }

  calculateConfidence(data, condition, prediction) {
    let base = 85;
    if (condition === prediction.level) base += 5;
    if (data.ambulance_queue > 0 && data.patient_inflow === 'HIGH') base += 6;
    if (data.average_wait_minutes > 15) base += 2;
    return Math.min(Math.max(base, 70), 98);
  }

  async saveDecision(decisionObj) {
    try {
      const checkQuery = `
        SELECT id FROM hospital_agent_decisions 
        WHERE hospital_id = $1 
        AND decision_type = $2 
        AND status = 'ACTIVE' 
        AND created_at > NOW() - INTERVAL '15 minutes'
        LIMIT 1
      `;
      const checkRes = await pool.query(checkQuery, [decisionObj.hospital_id, decisionObj.decision_type]);
      
      if (checkRes.rows.length > 0) return decisionObj; 

      const query = `
        INSERT INTO hospital_agent_decisions 
        (hospital_id, decision_type, situation, decision, recommendation, expected_impact, confidence, status)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING *;
      `;
      const values = [
        decisionObj.hospital_id, decisionObj.decision_type, decisionObj.reason,
        decisionObj.decision, decisionObj.recommendation, decisionObj.expected_impact,
        decisionObj.confidence, 'ACTIVE'
      ];
      
      const res = await pool.query(query, values);
      return { ...decisionObj, db_id: res.rows[0].id };
    } catch (error) {
      console.error('[HOSPITAL AGENT] Failed to persist decision:', error);
      return decisionObj; 
    }
  }

  async evaluate(event) {
    if (event.type === 'ACCIDENT_DETECTED' || event.type === 'MEDICAL_EMERGENCY') {
      return this._createDecision(
        'Advisory',
        event.severity,
        'AMBULANCE_RESPONSE_RECOMMENDED',
        0.95,
        { recommendedAction: 'Dispatch emergency transport if verified by Police/Traffic agents.' },
        'Trauma event reported in regional network.',
        ['Traffic Delays'],
        ['TrafficAgent', 'PoliceAgent']
      );
    }
    
    return this._createDecision(
      'Monitoring',
      event.severity || 'Low',
      'CAPACITY_MONITORING_ACTIVE',
      0.90,
      {},
      `Monitoring hospital capacity in response to ${event.type}.`,
      [],
      []
    );
  }
}

module.exports = HospitalAgent;
