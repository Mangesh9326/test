--
-- PostgreSQL database dump
--

\restrict a2OLf9Fs0kuQn1ENds0ESQStfLEA8CNUnA95GWe3UBRlLXiVNdrxu1UqNL9uc8U

-- Dumped from database version 18.4
-- Dumped by pg_dump version 18.4

-- Started on 2026-09-08 13:50:44

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 241 (class 1259 OID 34041)
-- Name: traffic_demo_scenarios; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.traffic_demo_scenarios (
    id bigint NOT NULL,
    name character varying(150) NOT NULL,
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
    source character varying(50) DEFAULT 'DEMONSTRATION'::character varying NOT NULL,
    is_demo boolean DEFAULT true NOT NULL,
    event_data jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT traffic_demo_scenarios_demonstration_check CHECK (((is_demo = true) AND ((source)::text = 'DEMONSTRATION'::text)))
);


ALTER TABLE public.traffic_demo_scenarios OWNER TO postgres;

--
-- TOC entry 240 (class 1259 OID 34040)
-- Name: traffic_demo_scenarios_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.traffic_demo_scenarios_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.traffic_demo_scenarios_id_seq OWNER TO postgres;

--
-- TOC entry 5069 (class 0 OID 0)
-- Dependencies: 240
-- Name: traffic_demo_scenarios_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.traffic_demo_scenarios_id_seq OWNED BY public.traffic_demo_scenarios.id;


--
-- TOC entry 4902 (class 2604 OID 34044)
-- Name: traffic_demo_scenarios id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_demo_scenarios ALTER COLUMN id SET DEFAULT nextval('public.traffic_demo_scenarios_id_seq'::regclass);


--
-- TOC entry 5062 (class 0 OID 34041)
-- Dependencies: 241
-- Data for Name: traffic_demo_scenarios; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.traffic_demo_scenarios (id, name, type, location, latitude, longitude, severity, description, traffic_level, density_level, flow_level, affected_approach, source, is_demo, event_data, created_at, updated_at) FROM stdin;
1	Dadar Peak Congestion	TRAFFIC_CONGESTION	Dadar Junction	19.0189000	72.8437000	SEVERE	During evening peak hours, traffic demand increases at Dadar Junction. The Traffic Agent detects deteriorating traffic conditions, forecasts further congestion, evaluates signal timing, and recommends a lower-impact route toward Wadala.	SEVERE	HIGH	HIGH	Dadar → Wadala	DEMONSTRATION	t	{"type": "TRAFFIC_CONDITION", "routes": [{"id": "dadar-wadala-main", "name": "Dadar–Wadala Main Road", "impact": "HIGH", "recommended": false, "trafficLevel": "SEVERE", "predictedTrafficLevel": "CRITICAL"}, {"id": "dadar-matunga-link", "name": "Matunga Link Road", "impact": "LOW", "recommended": true, "trafficLevel": "HIGH", "predictedTrafficLevel": "HIGH"}], "signal": {"reason": "High traffic demand on the Dadar to Wadala approach.", "status": "SIMULATED", "current": {"red": 48, "green": 42}, "signalId": "DADAR-WADALA-01", "cycleTime": 90, "adjustment": {"red": -8, "green": 8}, "recommended": {"red": 40, "green": 50}, "intersection": "Dadar Junction"}, "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "metadata": {"source": "DEMONSTRATION", "flowLevel": "HIGH", "densityLevel": "HIGH", "signalDemand": "SEVERE", "trafficLevel": "SEVERE", "roadCondition": "NORMAL", "affectedApproach": "Dadar → Wadala"}, "severity": "SEVERE", "agentPlan": {"actions": ["ANALYZE_TRAFFIC", "PREDICT_CONGESTION", "OPTIMIZE_SIGNAL", "EVALUATE_ROUTES", "GENERATE_RECOMMENDATION"], "primaryAgent": "TrafficAgent", "primaryAction": "SIGNAL_OPTIMIZATION", "affectedAgents": ["TrafficAgent"]}, "demoConfig": {"autoRoute": true, "autoAnalyze": true, "defaultOrigin": "Dadar Station, Mumbai", "showAgentLogs": true, "showPrediction": true, "showTrafficLayer": true, "defaultDestination": "Wadala, Mumbai", "showSignalOptimization": true}, "prediction": {"type": "CONGESTION", "source": "DEMONSTRATION", "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "scenarioSteps": [{"step": 1, "type": "DETECT", "message": "Severe traffic conditions detected at Dadar Junction."}, {"step": 2, "type": "ANALYZE", "message": "Traffic Agent is analyzing demand on the Dadar to Wadala approach."}, {"step": 3, "type": "PREDICT", "message": "Further congestion is expected within the next 15 minutes."}, {"step": 4, "type": "SIGNAL_OPTIMIZATION", "message": "Traffic Agent recommends increasing the green phase by 8 seconds."}, {"step": 5, "type": "ROUTE_EVALUATION", "message": "Alternative routes toward Wadala are being evaluated."}, {"step": 6, "type": "DECISION", "message": "Matunga Link Road is recommended as the lower-impact alternative."}], "expectedOutcome": {"status": "SIMULATED", "redAfter": 40, "redBefore": 48, "greenAfter": 50, "greenBefore": 42, "trafficAfter": "IMPROVING", "trafficBefore": "SEVERE", "routeRecommendation": "Matunga Link Road"}, "scenarioVersion": "2.0", "densityPrediction": {"source": "DEMONSTRATION", "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "VERY_HIGH"}}	2026-09-01 21:39:50.36267	2026-09-01 23:04:01.292935
2	Dadar Accident Route Diversion	ACCIDENT_DETECTED	Dadar–Wadala corridor	19.0231000	72.8534000	HIGH	A simulated road collision partially blocks the Dadar–Wadala corridor. The Traffic Agent evaluates the affected road, predicts secondary congestion, recommends a diversion, and coordinates simulated traffic management.	HIGH	HIGH	MODERATE	Dadar → Wadala	DEMONSTRATION	t	{"type": "ACCIDENT_DETECTED", "routes": [{"id": "accident-main", "name": "Dadar–Wadala Main Road", "impact": "CRITICAL", "roadClosed": false, "recommended": false, "trafficLevel": "SEVERE", "incidentImpact": true, "predictedTrafficLevel": "CRITICAL"}, {"id": "accident-alt", "name": "Matunga Link Road", "impact": "MODERATE", "roadClosed": false, "recommended": true, "trafficLevel": "HIGH", "incidentImpact": false, "predictedTrafficLevel": "HIGH"}], "signal": {"reason": "Reduce inflow toward the partially blocked corridor.", "status": "SIMULATED", "current": {"red": 50, "green": 40}, "signalId": "DADAR-WADALA-01", "cycleTime": 90, "recommended": {"red": 56, "green": 34}, "intersection": "Dadar Junction"}, "incident": {"injuries": "SIMULATED_UNKNOWN", "roadBlocked": true, "incidentType": "ROAD_ACCIDENT", "lanesAffected": 1, "vehiclesInvolved": 2}, "location": {"city": "Mumbai", "name": "Dadar–Wadala corridor", "latitude": 19.0231, "longitude": 72.8534}, "metadata": {"source": "DEMONSTRATION", "flowLevel": "MODERATE", "densityLevel": "HIGH", "signalDemand": "HIGH", "trafficLevel": "HIGH", "roadCondition": "PARTIALLY_BLOCKED", "affectedApproach": "Dadar → Wadala"}, "severity": "HIGH", "agentPlan": {"actions": ["ASSESS_ACCIDENT", "IDENTIFY_AFFECTED_CORRIDOR", "PREDICT_SECONDARY_CONGESTION", "EVALUATE_ROUTES", "OPTIMIZE_SIGNAL", "GENERATE_DIVERSION"], "primaryAgent": "TrafficAgent", "primaryAction": "ROUTE_DIVERSION", "affectedAgents": ["TrafficAgent", "PoliceAgent"]}, "demoConfig": {"autoRoute": true, "autoAnalyze": true, "showIncident": true, "defaultOrigin": "Dadar Station, Mumbai", "showAgentLogs": true, "showTrafficLayer": true, "defaultDestination": "Wadala, Mumbai", "showSignalOptimization": true}, "prediction": {"type": "SECONDARY_CONGESTION", "source": "DEMONSTRATION", "currentState": "HIGH", "horizonMinutes": 10, "predictedState": "SEVERE"}, "scenarioSteps": [{"step": 1, "type": "DETECT", "message": "Road accident detected on the Dadar–Wadala corridor."}, {"step": 2, "type": "IMPACT_ANALYSIS", "message": "Traffic Agent is assessing the blocked lane and downstream impact."}, {"step": 3, "type": "PREDICT", "message": "Secondary congestion is expected to increase around the affected corridor."}, {"step": 4, "type": "ROUTE_EVALUATION", "message": "Alternative routes are being evaluated."}, {"step": 5, "type": "SIGNAL_OPTIMIZATION", "message": "Signal timing adjustment is recommended to reduce additional inflow."}, {"step": 6, "type": "DECISION", "message": "Traffic is recommended to divert toward Matunga Link Road."}], "expectedOutcome": {"status": "SIMULATED", "affectedRoute": "Dadar–Wadala Main Road", "recommendedRoute": "Matunga Link Road", "signalAdjustment": "Reduce inflow toward affected corridor"}, "scenarioVersion": "2.0"}	2026-09-01 21:39:50.36267	2026-09-01 23:04:01.292935
4	Wadala Road Closure	ROAD_CLOSURE	Wadala	19.0176000	72.8580000	HIGH	A simulated road closure makes a key Wadala road unavailable. The Traffic Agent identifies the affected corridor, evaluates alternatives, and recommends a diversion.	HIGH	HIGH	LOW	Wadala Junction	DEMONSTRATION	t	{"type": "ROAD_CLOSURE", "routes": [{"id": "wadala-closed", "name": "Wadala Main Road", "impact": "CRITICAL", "roadClosed": true, "recommended": false, "trafficLevel": "CRITICAL", "predictedTrafficLevel": "CRITICAL"}, {"id": "wadala-alt", "name": "Sion–Matunga Link", "impact": "LOW", "roadClosed": false, "recommended": true, "trafficLevel": "MODERATE", "predictedTrafficLevel": "HIGH"}], "location": {"city": "Mumbai", "name": "Wadala", "latitude": 19.0176, "longitude": 72.8580}, "metadata": {"source": "DEMONSTRATION", "flowLevel": "LOW", "roadClosed": true, "densityLevel": "HIGH", "trafficLevel": "HIGH", "roadCondition": "CLOSED", "affectedApproach": "Wadala Junction"}, "severity": "HIGH", "agentPlan": {"actions": ["DETECT_CLOSURE", "IDENTIFY_AFFECTED_ROAD", "EVALUATE_ALTERNATIVES", "RECOMMEND_DIVERSION"], "primaryAgent": "TrafficAgent", "primaryAction": "ROAD_DIVERSION", "affectedAgents": ["TrafficAgent", "CitizenAgent"]}, "demoConfig": {"autoRoute": true, "autoAnalyze": true, "defaultOrigin": "Dadar Station, Mumbai", "showAgentLogs": true, "showRoadClosure": true, "defaultDestination": "Wadala, Mumbai", "showRouteDiversion": true}, "roadClosure": {"road": "Wadala Main Road", "reason": "Infrastructure maintenance demonstration", "status": "SIMULATED_CLOSED", "estimatedDurationMinutes": 45}, "scenarioSteps": [{"step": 1, "type": "DETECT", "message": "Road closure detected in the Wadala corridor."}, {"step": 2, "type": "IMPACT_ANALYSIS", "message": "Traffic Agent identified the primary route as unavailable."}, {"step": 3, "type": "ROUTE_EVALUATION", "message": "Alternative routes are being evaluated."}, {"step": 4, "type": "DECISION", "message": "Sion–Matunga Link is recommended as the alternative route."}], "expectedOutcome": {"status": "SIMULATED", "closedRoad": "Wadala Main Road", "recommendedRoute": "Sion–Matunga Link"}, "scenarioVersion": "2.0"}	2026-09-01 21:39:50.36267	2026-09-01 23:04:01.292935
3	Ambulance Green Corridor	EMERGENCY_ROUTE	Dadar → Sion → KEM Hospital	19.0208000	72.8509000	CRITICAL	A simulated emergency ambulance route is requested from Dadar toward KEM Hospital. The Traffic Agent evaluates the corridor and generates simulated signal-priority recommendations at key intersections.	HIGH	HIGH	HIGH	Dadar → Sion → KEM Hospital	DEMONSTRATION	t	{"type": "EMERGENCY_ROUTE", "route": {"name": "Dadar → Sion → KEM Hospital", "waypoints": [{"name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, {"name": "Sion Circle", "latitude": 19.0460, "longitude": 72.8620}, {"name": "KEM Hospital", "latitude": 19.0015, "longitude": 72.8420}]}, "location": {"city": "Mumbai", "name": "Dadar → Sion → KEM Hospital"}, "metadata": {"source": "DEMONSTRATION", "flowLevel": "HIGH", "simulation": true, "densityLevel": "HIGH", "trafficLevel": "HIGH", "affectedApproach": "Dadar → Sion → KEM Hospital"}, "severity": "CRITICAL", "agentPlan": {"actions": ["RECEIVE_EMERGENCY_REQUEST", "EVALUATE_ROUTE", "IDENTIFY_INTERSECTIONS", "GENERATE_SIGNAL_PRIORITY", "MONITOR_CORRIDOR"], "primaryAgent": "TrafficAgent", "primaryAction": "EMERGENCY_PRIORITY", "affectedAgents": ["TrafficAgent", "HospitalAgent"]}, "emergency": {"type": "AMBULANCE", "origin": "Dadar Station, Mumbai", "priority": "CRITICAL", "vehicleId": "AMB-DEMO-104", "destination": "KEM Hospital, Mumbai"}, "demoConfig": {"autoRoute": true, "autoAnalyze": true, "defaultOrigin": "Dadar Station, Mumbai", "showAgentLogs": true, "defaultDestination": "KEM Hospital, Mumbai", "showEmergencyRoute": true, "showSignalPriority": true}, "intersections": [{"name": "Dadar Junction", "priority": "SIMULATED", "signalId": "DADAR-EMERGENCY-01"}, {"name": "Sion Circle", "priority": "SIMULATED", "signalId": "SION-EMERGENCY-01"}], "scenarioSteps": [{"step": 1, "type": "EMERGENCY", "message": "Emergency ambulance priority request received."}, {"step": 2, "type": "ROUTE", "message": "Traffic Agent is evaluating the Dadar to KEM Hospital corridor."}, {"step": 3, "type": "INTERSECTION_ANALYSIS", "message": "Key intersections along the emergency corridor identified."}, {"step": 4, "type": "SIGNAL_PRIORITY", "message": "Simulated signal-priority recommendations generated."}, {"step": 5, "type": "CORRIDOR", "message": "Green corridor coordination is active in simulation mode."}], "expectedOutcome": {"status": "SIMULATED", "priority": "EMERGENCY", "corridorStatus": "SIMULATED_ACTIVE", "intersectionsOptimized": 2}, "scenarioVersion": "2.0"}	2026-09-01 21:39:50.36267	2026-09-01 23:04:01.292935
5	BKC Event Traffic Surge	TRAFFIC_PREDICTION	BKC	19.0609000	72.8686000	HIGH	A simulated major event in BKC is expected to increase traffic demand. The Traffic Agent forecasts congestion and generates proactive signal and route recommendations before conditions become critical.	HIGH	HIGH	HIGH	BKC entry corridors	DEMONSTRATION	t	{"type": "TRAFFIC_PREDICTION", "event": {"type": "LARGE_EVENT", "location": "BKC", "expectedDemand": "HIGH", "planningHorizonMinutes": 30}, "routes": [{"id": "bkc-west", "name": "Bandra–Kurla Connector", "impact": "HIGH", "recommended": false, "trafficLevel": "HIGH", "predictedTrafficLevel": "SEVERE"}, {"id": "bkc-east", "name": "Sion–Dharavi Link", "impact": "MODERATE", "recommended": true, "trafficLevel": "MODERATE", "predictedTrafficLevel": "HIGH"}], "signal": {"reason": "Expected increase in event-related traffic demand.", "status": "SIMULATED", "current": {"red": 45, "green": 45}, "signalId": "BKC-EAST-01", "cycleTime": 90, "adjustment": {"red": -8, "green": 8}, "recommended": {"red": 37, "green": 53}, "intersection": "BKC Entry"}, "location": {"city": "Mumbai", "name": "BKC", "latitude": 19.0609, "longitude": 72.8686}, "metadata": {"source": "DEMONSTRATION", "flowLevel": "HIGH", "densityLevel": "HIGH", "signalDemand": "HIGH", "trafficLevel": "HIGH", "affectedApproach": "BKC entry corridors"}, "severity": "HIGH", "agentPlan": {"actions": ["ANALYZE_EVENT_DEMAND", "PREDICT_CONGESTION", "OPTIMIZE_SIGNAL", "EVALUATE_ROUTES", "GENERATE_PREVENTIVE_RECOMMENDATION"], "primaryAgent": "TrafficAgent", "primaryAction": "PROACTIVE_OPTIMIZATION", "affectedAgents": ["TrafficAgent", "CitizenAgent"]}, "demoConfig": {"autoRoute": true, "autoAnalyze": true, "defaultOrigin": "Bandra, Mumbai", "showAgentLogs": true, "showPrediction": true, "defaultDestination": "BKC, Mumbai", "showSignalOptimization": true}, "prediction": {"type": "TRAFFIC_SURGE", "source": "DEMONSTRATION", "currentState": "HIGH", "horizonMinutes": 30, "predictedState": "SEVERE"}, "scenarioSteps": [{"step": 1, "type": "PREDICTION", "message": "Increased traffic demand is expected around BKC."}, {"step": 2, "type": "ANALYZE", "message": "Traffic Agent is analyzing expected event-related demand."}, {"step": 3, "type": "PREDICT", "message": "Traffic is predicted to reach severe levels within 30 minutes."}, {"step": 4, "type": "SIGNAL_OPTIMIZATION", "message": "Proactive green-phase extension is recommended."}, {"step": 5, "type": "ROUTE_EVALUATION", "message": "Alternative entry corridors are being evaluated."}, {"step": 6, "type": "DECISION", "message": "Sion–Dharavi Link is recommended to distribute traffic demand."}], "expectedOutcome": {"status": "SIMULATED", "trafficBefore": "HIGH", "predictedTraffic": "SEVERE", "recommendedRoute": "Sion–Dharavi Link", "signalAdjustment": "+8 seconds green"}, "scenarioVersion": "2.0", "densityPrediction": {"source": "DEMONSTRATION", "currentState": "HIGH", "horizonMinutes": 30, "predictedState": "VERY_HIGH"}}	2026-09-01 21:39:50.36267	2026-09-01 23:04:01.292935
6	Monsoon Waterlogging	FLOOD	Sion / Wadala	19.0351000	72.8629000	HIGH	Simulated monsoon waterlogging affects a low-lying road between Sion and Wadala. The Traffic Agent identifies the unsafe corridor, recommends restricting the route, and evaluates an alternative diversion.	HIGH	HIGH	LOW	Sion / Wadala low-lying approach	DEMONSTRATION	t	{"type": "FLOOD", "routes": [{"id": "flooded-road", "name": "Sion–Wadala low-lying road", "impact": "CRITICAL", "roadClosed": true, "recommended": false, "trafficLevel": "CRITICAL", "predictedTrafficLevel": "CRITICAL"}, {"id": "flood-alt", "name": "Matunga diversion", "impact": "MODERATE", "roadClosed": false, "recommended": true, "trafficLevel": "MODERATE", "predictedTrafficLevel": "HIGH"}], "location": {"city": "Mumbai", "name": "Sion / Wadala", "latitude": 19.0351, "longitude": 72.8629}, "metadata": {"source": "DEMONSTRATION", "flowLevel": "LOW", "densityLevel": "HIGH", "trafficLevel": "HIGH", "roadCondition": "POOR", "affectedApproach": "Sion / Wadala low-lying approach"}, "severity": "HIGH", "agentPlan": {"actions": ["ASSESS_ROAD_CONDITION", "IDENTIFY_UNSAFE_CORRIDOR", "RESTRICT_ROUTE", "EVALUATE_ALTERNATIVES", "RECOMMEND_DIVERSION"], "primaryAgent": "TrafficAgent", "primaryAction": "ROAD_RESTRICTION_AND_DIVERSION", "affectedAgents": ["TrafficAgent", "CitizenAgent", "EnvironmentalAgent"]}, "demoConfig": {"autoRoute": true, "autoAnalyze": true, "defaultOrigin": "Sion, Mumbai", "showAgentLogs": true, "defaultDestination": "Wadala, Mumbai", "showRouteDiversion": true, "showRoadRestriction": true}, "environment": {"condition": "HEAVY_RAIN", "roadSafety": "DEGRADED", "waterlogging": true}, "scenarioSteps": [{"step": 1, "type": "DETECT", "message": "Waterlogging detected on the Sion–Wadala low-lying approach."}, {"step": 2, "type": "ASSESS", "message": "Traffic Agent is assessing road safety and traffic impact."}, {"step": 3, "type": "RESTRICTION", "message": "Simulated route restriction recommended for the affected road."}, {"step": 4, "type": "ROUTE_EVALUATION", "message": "Alternative routes are being evaluated."}, {"step": 5, "type": "DECISION", "message": "Matunga diversion is recommended."}], "expectedOutcome": {"status": "SIMULATED", "restrictedRoad": "Sion–Wadala low-lying road", "recommendedRoute": "Matunga diversion"}, "roadRestriction": {"road": "Sion–Wadala low-lying road", "reason": "Waterlogging", "status": "SIMULATED_RESTRICTION", "estimatedDurationMinutes": 60}, "scenarioVersion": "2.0"}	2026-09-01 21:39:50.36267	2026-09-01 23:04:01.292935
\.


--
-- TOC entry 5071 (class 0 OID 0)
-- Dependencies: 240
-- Name: traffic_demo_scenarios_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.traffic_demo_scenarios_id_seq', 6, true);


--
-- TOC entry 4911 (class 2606 OID 34069)
-- Name: traffic_demo_scenarios traffic_demo_scenarios_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_demo_scenarios
    ADD CONSTRAINT traffic_demo_scenarios_name_key UNIQUE (name);


--
-- TOC entry 4913 (class 2606 OID 34067)
-- Name: traffic_demo_scenarios traffic_demo_scenarios_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_demo_scenarios
    ADD CONSTRAINT traffic_demo_scenarios_pkey PRIMARY KEY (id);


--
-- TOC entry 4909 (class 1259 OID 34070)
-- Name: idx_traffic_demo_scenarios_demo; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_traffic_demo_scenarios_demo ON public.traffic_demo_scenarios USING btree (is_demo, name);


--
-- TOC entry 5068 (class 0 OID 0)
-- Dependencies: 241
-- Name: TABLE traffic_demo_scenarios; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.traffic_demo_scenarios TO smart_city_user;


--
-- TOC entry 5070 (class 0 OID 0)
-- Dependencies: 240
-- Name: SEQUENCE traffic_demo_scenarios_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.traffic_demo_scenarios_id_seq TO smart_city_user;


-- Completed on 2026-09-08 13:50:44

--
-- PostgreSQL database dump complete
--

\unrestrict a2OLf9Fs0kuQn1ENds0ESQStfLEA8CNUnA95GWe3UBRlLXiVNdrxu1UqNL9uc8U

