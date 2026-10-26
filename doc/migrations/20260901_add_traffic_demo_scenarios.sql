-- Database-backed Mumbai traffic demonstration scenarios.
CREATE TABLE IF NOT EXISTS public.traffic_demo_scenarios (
    id bigserial PRIMARY KEY,
    name character varying(150) NOT NULL UNIQUE,
    type character varying(80) NOT NULL,
    location character varying(255) NOT NULL,
    latitude numeric(10,7) NOT NULL,
    longitude numeric(10,7) NOT NULL,
    severity character varying(20) NOT NULL,
    description text NOT NULL,
    traffic_level character varying(20),
    density_level character varying(20),
    flow_level character varying(20),
    affected_approach character varying(255),
    source character varying(50) NOT NULL DEFAULT 'DEMONSTRATION',
    is_demo boolean NOT NULL DEFAULT true,
    event_data jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT traffic_demo_scenarios_demonstration_check CHECK (is_demo = true AND source = 'DEMONSTRATION')
);

CREATE INDEX IF NOT EXISTS idx_traffic_demo_scenarios_demo ON public.traffic_demo_scenarios (is_demo, name);

INSERT INTO public.traffic_demo_scenarios
    (name, type, location, latitude, longitude, severity, description, traffic_level, density_level, flow_level, affected_approach, source, is_demo, event_data)
VALUES
('Dadar Peak Congestion', 'TRAFFIC_CONGESTION', 'Dadar Junction', 19.0189000, 72.8437000, 'SEVERE', 'Evening peak demand is increasing on the Dadar to Wadala approach. The agent predicts worsening congestion, evaluates simulated signal timing, and recommends the lower-impact route.', 'SEVERE', 'HIGH', 'HIGH', 'Dadar → Wadala', 'DEMONSTRATION', true,
 '{"type":"TRAFFIC_CONDITION","location":"Dadar Junction","severity":"SEVERE","metadata":{"trafficLevel":"SEVERE","densityLevel":"HIGH","flowLevel":"HIGH","signalDemand":"SEVERE","affectedApproach":"Dadar → Wadala","signalId":"DADAR-WADALA-01","currentGreenTime":42,"currentRedTime":48,"source":"DEMONSTRATION","prediction":{"predictedState":"CRITICAL","horizonMinutes":15,"source":"DEMONSTRATION"},"routes":[{"id":"dadar-wadala-main","name":"Dadar–Wadala Main Road","trafficLevel":"SEVERE","predictedTrafficLevel":"CRITICAL"},{"id":"dadar-matunga-link","name":"Matunga Link Road","trafficLevel":"HIGH","predictedTrafficLevel":"HIGH"}]}}'::jsonb),
('Dadar Accident Route Diversion', 'ACCIDENT_DETECTED', 'Dadar–Wadala corridor', 19.0231000, 72.8534000, 'HIGH', 'A collision partially blocks the Dadar–Wadala corridor. The agent assesses the impact, recommends diversion, and evaluates a simulated signal adjustment.', 'HIGH', 'HIGH', 'MODERATE', 'Dadar → Wadala', 'DEMONSTRATION', true,
 '{"type":"ACCIDENT_DETECTED","location":"Dadar–Wadala corridor","severity":"HIGH","metadata":{"vehicles":["Car","Bus"],"trafficLevel":"HIGH","densityLevel":"HIGH","flowLevel":"MODERATE","signalDemand":"HIGH","affectedApproach":"Dadar → Wadala","currentGreenTime":40,"currentRedTime":50,"source":"DEMONSTRATION","routes":[{"id":"accident-main","name":"Dadar–Wadala Main Road","trafficLevel":"SEVERE","predictedTrafficLevel":"CRITICAL","incidentImpact":true},{"id":"accident-alt","name":"Matunga Link Road","trafficLevel":"HIGH","predictedTrafficLevel":"HIGH"}]}}'::jsonb),
('Ambulance Green Corridor', 'EMERGENCY_ROUTE', 'Dadar → Sion → KEM Hospital', 19.0208000, 72.8509000, 'CRITICAL', 'A simulated ambulance priority request from Dadar through Sion to KEM Hospital. The recommendation is a simulation only and does not control real Mumbai signals.', 'HIGH', 'HIGH', 'HIGH', 'Dadar → Sion → KEM Hospital', 'DEMONSTRATION', true,
 '{"type":"EMERGENCY_ROUTE","location":"Dadar → Sion → KEM Hospital","severity":"CRITICAL","metadata":{"trafficLevel":"HIGH","densityLevel":"HIGH","flowLevel":"HIGH","affectedApproach":"Dadar → Sion → KEM Hospital","intersections":["Dadar Junction","Sion Circle","KEM Hospital approach"],"currentGreenTime":38,"currentRedTime":52,"source":"DEMONSTRATION","simulation":true}}'::jsonb),
('Wadala Road Closure', 'ROAD_CLOSURE', 'Wadala', 19.0176000, 72.8580000, 'HIGH', 'A Wadala road section is unavailable. The agent identifies affected routes and recommends a diversion around the highlighted closure area.', 'HIGH', 'HIGH', 'LOW', 'Wadala Junction', 'DEMONSTRATION', true,
 '{"type":"ROAD_CLOSURE","location":"Wadala","severity":"HIGH","metadata":{"trafficLevel":"HIGH","densityLevel":"HIGH","flowLevel":"LOW","roadClosed":true,"affectedApproach":"Wadala Junction","source":"DEMONSTRATION","routes":[{"id":"wadala-closed","name":"Wadala Main Road","trafficLevel":"CRITICAL","roadClosed":true},{"id":"wadala-alt","name":"Sion–Matunga Link","trafficLevel":"MODERATE"}]}}'::jsonb),
('BKC Event Traffic Surge', 'TRAFFIC_PREDICTION', 'BKC', 19.0609000, 72.8686000, 'HIGH', 'A large BKC event is expected to increase demand. The agent forecasts deterioration and recommends proactive simulated signal optimisation and alternate routing.', 'HIGH', 'HIGH', 'HIGH', 'BKC entry corridors', 'DEMONSTRATION', true,
 '{"type":"TRAFFIC_PREDICTION","location":"BKC","severity":"HIGH","metadata":{"trafficLevel":"HIGH","densityLevel":"HIGH","flowLevel":"HIGH","signalDemand":"HIGH","affectedApproach":"BKC entry corridors","currentGreenTime":45,"currentRedTime":45,"source":"DEMONSTRATION","prediction":{"predictedState":"SEVERE","horizonMinutes":30,"source":"DEMONSTRATION"},"routes":[{"id":"bkc-west","name":"Bandra–Kurla Connector","trafficLevel":"HIGH","predictedTrafficLevel":"SEVERE"},{"id":"bkc-east","name":"Sion–Dharavi Link","trafficLevel":"MODERATE","predictedTrafficLevel":"HIGH"}]}}'::jsonb),
('Monsoon Waterlogging', 'FLOOD', 'Sion / Wadala', 19.0351000, 72.8629000, 'HIGH', 'Heavy rainfall causes waterlogging between Sion and Wadala. The agent recommends avoiding the affected road and evaluates an alternative route with a simulated restriction.', 'HIGH', 'HIGH', 'LOW', 'Sion / Wadala low-lying approach', 'DEMONSTRATION', true,
 '{"type":"FLOOD","location":"Sion / Wadala","severity":"HIGH","metadata":{"trafficLevel":"HIGH","densityLevel":"HIGH","flowLevel":"LOW","roadCondition":"POOR","affectedApproach":"Sion / Wadala low-lying approach","source":"DEMONSTRATION","routes":[{"id":"flooded-road","name":"Sion–Wadala low-lying road","trafficLevel":"CRITICAL","roadClosed":true},{"id":"flood-alt","name":"Matunga diversion","trafficLevel":"MODERATE"}]}}'::jsonb)
ON CONFLICT (name) DO UPDATE SET
    type = EXCLUDED.type, location = EXCLUDED.location, latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude, severity = EXCLUDED.severity, description = EXCLUDED.description,
    traffic_level = EXCLUDED.traffic_level, density_level = EXCLUDED.density_level,
    flow_level = EXCLUDED.flow_level, affected_approach = EXCLUDED.affected_approach,
    source = EXCLUDED.source, is_demo = EXCLUDED.is_demo, event_data = EXCLUDED.event_data,
    updated_at = CURRENT_TIMESTAMP;
