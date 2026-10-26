--
-- PostgreSQL database dump
--

\restrict 8yqyVAJdsgjbf43hQDfmMs3yWGMu09e1LpCTXeeLgnga06PevIEpBtMaEq3hEdC

-- Dumped from database version 18.4
-- Dumped by pg_dump version 18.4

-- Started on 2026-10-02 15:55:23

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

--
-- TOC entry 2 (class 3079 OID 34122)
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- TOC entry 5326 (class 0 OID 0)
-- Dependencies: 2
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 253 (class 1259 OID 34265)
-- Name: ambulance_operations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ambulance_operations (
    id bigint NOT NULL,
    ambulance_id character varying(50) NOT NULL,
    incident_id character varying(100),
    current_latitude numeric(10,6),
    current_longitude numeric(10,6),
    destination_hospital_id uuid,
    priority character varying(50) DEFAULT 'STANDARD'::character varying,
    eta_minutes integer,
    status character varying(50) DEFAULT 'EN_ROUTE'::character varying,
    last_updated timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.ambulance_operations OWNER TO postgres;

--
-- TOC entry 252 (class 1259 OID 34264)
-- Name: ambulance_operations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ambulance_operations_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.ambulance_operations_id_seq OWNER TO postgres;

--
-- TOC entry 5328 (class 0 OID 0)
-- Dependencies: 252
-- Name: ambulance_operations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ambulance_operations_id_seq OWNED BY public.ambulance_operations.id;


--
-- TOC entry 249 (class 1259 OID 34216)
-- Name: hospital_agent_decisions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hospital_agent_decisions (
    id bigint NOT NULL,
    hospital_id uuid NOT NULL,
    decision_type character varying(100) NOT NULL,
    situation text,
    decision text,
    recommendation text,
    expected_impact text,
    confidence numeric(5,2),
    status character varying(50) DEFAULT 'PENDING'::character varying,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.hospital_agent_decisions OWNER TO postgres;

--
-- TOC entry 248 (class 1259 OID 34215)
-- Name: hospital_agent_decisions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.hospital_agent_decisions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hospital_agent_decisions_id_seq OWNER TO postgres;

--
-- TOC entry 5331 (class 0 OID 0)
-- Dependencies: 248
-- Name: hospital_agent_decisions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.hospital_agent_decisions_id_seq OWNED BY public.hospital_agent_decisions.id;


--
-- TOC entry 247 (class 1259 OID 34179)
-- Name: hospital_monitoring; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hospital_monitoring (
    id bigint NOT NULL,
    hospital_id uuid NOT NULL,
    total_beds integer NOT NULL,
    occupied_beds integer NOT NULL,
    available_beds integer NOT NULL,
    icu_total integer NOT NULL,
    icu_occupied integer NOT NULL,
    icu_available integer NOT NULL,
    emergency_capacity integer NOT NULL,
    emergency_occupancy integer NOT NULL,
    ambulance_queue integer DEFAULT 0,
    average_wait_minutes integer DEFAULT 0,
    patient_inflow character varying(50) DEFAULT 'Normal'::character varying,
    predicted_inflow character varying(50) DEFAULT 'Normal'::character varying,
    capacity_level character varying(50) DEFAULT 'Normal'::character varying,
    emergency_level character varying(50) DEFAULT 'Normal'::character varying,
    data_source character varying(50) DEFAULT 'SIMULATED_TEST_DATA'::character varying,
    last_observed_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_available_beds CHECK ((available_beds = (total_beds - occupied_beds))),
    CONSTRAINT chk_icu_available CHECK ((icu_available = (icu_total - icu_occupied))),
    CONSTRAINT hospital_monitoring_icu_occupied_check CHECK ((icu_occupied >= 0)),
    CONSTRAINT hospital_monitoring_icu_total_check CHECK ((icu_total >= 0)),
    CONSTRAINT hospital_monitoring_occupied_beds_check CHECK ((occupied_beds >= 0)),
    CONSTRAINT hospital_monitoring_total_beds_check CHECK ((total_beds >= 0))
);


ALTER TABLE public.hospital_monitoring OWNER TO postgres;

--
-- TOC entry 246 (class 1259 OID 34178)
-- Name: hospital_monitoring_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.hospital_monitoring_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hospital_monitoring_id_seq OWNER TO postgres;

--
-- TOC entry 5334 (class 0 OID 0)
-- Dependencies: 246
-- Name: hospital_monitoring_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.hospital_monitoring_id_seq OWNED BY public.hospital_monitoring.id;


--
-- TOC entry 251 (class 1259 OID 34239)
-- Name: hospital_resources; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hospital_resources (
    id bigint NOT NULL,
    hospital_id uuid NOT NULL,
    resource_type character varying(100) NOT NULL,
    total_capacity numeric NOT NULL,
    in_use_qty numeric NOT NULL,
    available_qty numeric NOT NULL,
    threshold_percent numeric DEFAULT 85.0,
    status character varying(50) DEFAULT 'NORMAL'::character varying,
    last_updated timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_resource_qty CHECK ((available_qty = (total_capacity - in_use_qty)))
);


ALTER TABLE public.hospital_resources OWNER TO postgres;

--
-- TOC entry 250 (class 1259 OID 34238)
-- Name: hospital_resources_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.hospital_resources_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hospital_resources_id_seq OWNER TO postgres;

--
-- TOC entry 5337 (class 0 OID 0)
-- Dependencies: 250
-- Name: hospital_resources_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.hospital_resources_id_seq OWNED BY public.hospital_resources.id;


--
-- TOC entry 245 (class 1259 OID 34163)
-- Name: hospitals; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hospitals (
    hospital_id uuid NOT NULL,
    name character varying(255) NOT NULL,
    area character varying(100) NOT NULL,
    latitude numeric(10,6) NOT NULL,
    longitude numeric(10,6) NOT NULL,
    hospital_type character varying(50) DEFAULT 'General'::character varying,
    emergency_available boolean DEFAULT true,
    status character varying(50) DEFAULT 'Active'::character varying,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.hospitals OWNER TO postgres;

--
-- TOC entry 228 (class 1259 OID 25635)
-- Name: incidents; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.incidents (
    incident_id uuid DEFAULT gen_random_uuid() NOT NULL,
    incident_type character varying(50) NOT NULL,
    severity character varying(20) NOT NULL,
    latitude numeric(10,7) NOT NULL,
    longitude numeric(10,7) NOT NULL,
    status character varying(50) DEFAULT 'active'::character varying,
    assigned_departments jsonb,
    timeline_events jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    upload_id integer
);


ALTER TABLE public.incidents OWNER TO postgres;

--
-- TOC entry 232 (class 1259 OID 25680)
-- Name: live_detection_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.live_detection_results (
    id integer NOT NULL,
    upload_id integer,
    frame integer NOT NULL,
    object_detected character varying(100) NOT NULL,
    confidence numeric(5,2) NOT NULL,
    bounding_box jsonb,
    tracking_id character varying(50),
    timestamp_second integer NOT NULL
);


ALTER TABLE public.live_detection_results OWNER TO postgres;

--
-- TOC entry 231 (class 1259 OID 25679)
-- Name: live_detection_results_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.live_detection_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.live_detection_results_id_seq OWNER TO postgres;

--
-- TOC entry 5342 (class 0 OID 0)
-- Dependencies: 231
-- Name: live_detection_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.live_detection_results_id_seq OWNED BY public.live_detection_results.id;


--
-- TOC entry 234 (class 1259 OID 25699)
-- Name: live_event_timeline; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.live_event_timeline (
    id integer NOT NULL,
    upload_id integer,
    event_type character varying(100) NOT NULL,
    frame integer NOT NULL,
    timestamp_second integer NOT NULL,
    description text NOT NULL
);


ALTER TABLE public.live_event_timeline OWNER TO postgres;

--
-- TOC entry 233 (class 1259 OID 25698)
-- Name: live_event_timeline_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.live_event_timeline_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.live_event_timeline_id_seq OWNER TO postgres;

--
-- TOC entry 5345 (class 0 OID 0)
-- Dependencies: 233
-- Name: live_event_timeline_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.live_event_timeline_id_seq OWNED BY public.live_event_timeline.id;


--
-- TOC entry 238 (class 1259 OID 25735)
-- Name: live_processing_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.live_processing_logs (
    id integer NOT NULL,
    upload_id integer,
    log_stage character varying(100) NOT NULL,
    message text NOT NULL,
    logged_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.live_processing_logs OWNER TO postgres;

--
-- TOC entry 237 (class 1259 OID 25734)
-- Name: live_processing_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.live_processing_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.live_processing_logs_id_seq OWNER TO postgres;

--
-- TOC entry 5348 (class 0 OID 0)
-- Dependencies: 237
-- Name: live_processing_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.live_processing_logs_id_seq OWNED BY public.live_processing_logs.id;


--
-- TOC entry 236 (class 1259 OID 25719)
-- Name: live_tracking_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.live_tracking_results (
    id integer NOT NULL,
    upload_id integer,
    tracking_id character varying(50) NOT NULL,
    object_class character varying(100) NOT NULL,
    current_latitude numeric(10,7),
    current_longitude numeric(10,7),
    speed_kmh numeric(5,2),
    timestamp_second integer NOT NULL
);


ALTER TABLE public.live_tracking_results OWNER TO postgres;

--
-- TOC entry 235 (class 1259 OID 25718)
-- Name: live_tracking_results_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.live_tracking_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.live_tracking_results_id_seq OWNER TO postgres;

--
-- TOC entry 5351 (class 0 OID 0)
-- Dependencies: 235
-- Name: live_tracking_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.live_tracking_results_id_seq OWNED BY public.live_tracking_results.id;


--
-- TOC entry 230 (class 1259 OID 25664)
-- Name: live_video_uploads; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.live_video_uploads (
    id integer NOT NULL,
    filename character varying(255) NOT NULL,
    filepath text NOT NULL,
    camera_id character varying(50) NOT NULL,
    location character varying(255) NOT NULL,
    status character varying(50) DEFAULT 'Processing'::character varying,
    uploaded_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    video_hash character varying(64),
    duration_seconds numeric(10,2),
    fps integer,
    processing_started_at timestamp without time zone,
    processing_completed_at timestamp without time zone
);


ALTER TABLE public.live_video_uploads OWNER TO postgres;

--
-- TOC entry 229 (class 1259 OID 25663)
-- Name: live_video_uploads_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.live_video_uploads_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.live_video_uploads_id_seq OWNER TO postgres;

--
-- TOC entry 5354 (class 0 OID 0)
-- Dependencies: 229
-- Name: live_video_uploads_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.live_video_uploads_id_seq OWNED BY public.live_video_uploads.id;


--
-- TOC entry 226 (class 1259 OID 25603)
-- Name: map_entities; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.map_entities (
    entity_id uuid DEFAULT gen_random_uuid() NOT NULL,
    entity_type character varying(50) NOT NULL,
    name character varying(255) NOT NULL,
    latitude numeric(10,7) NOT NULL,
    longitude numeric(10,7) NOT NULL,
    metadata jsonb NOT NULL,
    layer_group character varying(50) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.map_entities OWNER TO postgres;

--
-- TOC entry 223 (class 1259 OID 25435)
-- Name: scenarios; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.scenarios (
    id integer NOT NULL,
    name character varying(100) NOT NULL,
    description text,
    duration_seconds integer NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.scenarios OWNER TO postgres;

--
-- TOC entry 222 (class 1259 OID 25434)
-- Name: scenarios_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.scenarios_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.scenarios_id_seq OWNER TO postgres;

--
-- TOC entry 5358 (class 0 OID 0)
-- Dependencies: 222
-- Name: scenarios_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.scenarios_id_seq OWNED BY public.scenarios.id;


--
-- TOC entry 225 (class 1259 OID 25448)
-- Name: timeline_events; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.timeline_events (
    id integer NOT NULL,
    scenario_id integer,
    timestamp_second integer NOT NULL,
    event_type character varying(50) NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.timeline_events OWNER TO postgres;

--
-- TOC entry 224 (class 1259 OID 25447)
-- Name: timeline_events_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.timeline_events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.timeline_events_id_seq OWNER TO postgres;

--
-- TOC entry 5361 (class 0 OID 0)
-- Dependencies: 224
-- Name: timeline_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.timeline_events_id_seq OWNED BY public.timeline_events.id;


--
-- TOC entry 240 (class 1259 OID 34023)
-- Name: traffic_decisions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.traffic_decisions (
    id integer NOT NULL,
    agent character varying(100) NOT NULL,
    decision_type character varying(100) NOT NULL,
    location character varying(255),
    severity character varying(50),
    recommendation text,
    reason text,
    payload jsonb,
    status character varying(50) DEFAULT 'RECOMMENDED'::character varying,
    source character varying(50) DEFAULT 'DEMONSTRATION'::character varying,
    outcome character varying(50) DEFAULT 'PENDING'::character varying,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.traffic_decisions OWNER TO postgres;

--
-- TOC entry 239 (class 1259 OID 34022)
-- Name: traffic_decisions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.traffic_decisions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.traffic_decisions_id_seq OWNER TO postgres;

--
-- TOC entry 5364 (class 0 OID 0)
-- Dependencies: 239
-- Name: traffic_decisions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.traffic_decisions_id_seq OWNED BY public.traffic_decisions.id;


--
-- TOC entry 242 (class 1259 OID 34041)
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
-- TOC entry 241 (class 1259 OID 34040)
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
-- TOC entry 5367 (class 0 OID 0)
-- Dependencies: 241
-- Name: traffic_demo_scenarios_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.traffic_demo_scenarios_id_seq OWNED BY public.traffic_demo_scenarios.id;


--
-- TOC entry 244 (class 1259 OID 34078)
-- Name: traffic_signal_monitoring; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.traffic_signal_monitoring (
    id bigint NOT NULL,
    entity_id uuid NOT NULL,
    signal_code character varying(50) NOT NULL,
    intersection_name character varying(150) NOT NULL,
    area character varying(100) NOT NULL,
    current_phase character varying(20) DEFAULT 'GREEN'::character varying NOT NULL,
    current_green_seconds integer DEFAULT 30 NOT NULL,
    current_red_seconds integer DEFAULT 30 NOT NULL,
    cycle_time_seconds integer DEFAULT 60 NOT NULL,
    traffic_level character varying(20) DEFAULT 'MODERATE'::character varying NOT NULL,
    density_level character varying(20) DEFAULT 'MODERATE'::character varying NOT NULL,
    affected_approach character varying(255),
    predicted_traffic_level character varying(20),
    predicted_density_level character varying(20),
    prediction_horizon_minutes integer DEFAULT 15,
    ai_decision character varying(100) DEFAULT 'Monitoring'::character varying,
    ai_recommendation character varying(255) DEFAULT 'No change'::character varying,
    recommended_green_seconds integer,
    recommended_red_seconds integer,
    green_adjustment_seconds integer DEFAULT 0,
    red_adjustment_seconds integer DEFAULT 0,
    expected_impact character varying(255),
    status character varying(50) DEFAULT 'MONITORING'::character varying,
    reason text,
    confidence numeric(5,2),
    last_observed_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.traffic_signal_monitoring OWNER TO postgres;

--
-- TOC entry 243 (class 1259 OID 34077)
-- Name: traffic_signal_monitoring_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.traffic_signal_monitoring_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.traffic_signal_monitoring_id_seq OWNER TO postgres;

--
-- TOC entry 5370 (class 0 OID 0)
-- Dependencies: 243
-- Name: traffic_signal_monitoring_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.traffic_signal_monitoring_id_seq OWNED BY public.traffic_signal_monitoring.id;


--
-- TOC entry 221 (class 1259 OID 25420)
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(50) NOT NULL,
    password_hash character varying(255) NOT NULL,
    role character varying(50) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT users_role_check CHECK (((role)::text = ANY ((ARRAY['Administrator'::character varying, 'Mayor'::character varying, 'Traffic Officer'::character varying, 'Police Officer'::character varying, 'Hospital Staff'::character varying, 'Fire Officer'::character varying, 'Utility Officer'::character varying, 'Citizen'::character varying])::text[])))
);


ALTER TABLE public.users OWNER TO postgres;

--
-- TOC entry 220 (class 1259 OID 25419)
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO postgres;

--
-- TOC entry 5373 (class 0 OID 0)
-- Dependencies: 220
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- TOC entry 227 (class 1259 OID 25619)
-- Name: vehicles; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vehicles (
    vehicle_id uuid DEFAULT gen_random_uuid() NOT NULL,
    vehicle_type character varying(50) NOT NULL,
    department character varying(50) NOT NULL,
    current_lat numeric(10,7) NOT NULL,
    current_lng numeric(10,7) NOT NULL,
    route_path jsonb NOT NULL,
    status character varying(50) DEFAULT 'patrolling'::character varying,
    last_updated timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.vehicles OWNER TO postgres;

--
-- TOC entry 5052 (class 2604 OID 34268)
-- Name: ambulance_operations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ambulance_operations ALTER COLUMN id SET DEFAULT nextval('public.ambulance_operations_id_seq'::regclass);


--
-- TOC entry 5045 (class 2604 OID 34219)
-- Name: hospital_agent_decisions id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_agent_decisions ALTER COLUMN id SET DEFAULT nextval('public.hospital_agent_decisions_id_seq'::regclass);


--
-- TOC entry 5035 (class 2604 OID 34182)
-- Name: hospital_monitoring id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_monitoring ALTER COLUMN id SET DEFAULT nextval('public.hospital_monitoring_id_seq'::regclass);


--
-- TOC entry 5048 (class 2604 OID 34242)
-- Name: hospital_resources id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_resources ALTER COLUMN id SET DEFAULT nextval('public.hospital_resources_id_seq'::regclass);


--
-- TOC entry 4999 (class 2604 OID 25683)
-- Name: live_detection_results id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_detection_results ALTER COLUMN id SET DEFAULT nextval('public.live_detection_results_id_seq'::regclass);


--
-- TOC entry 5000 (class 2604 OID 25702)
-- Name: live_event_timeline id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_event_timeline ALTER COLUMN id SET DEFAULT nextval('public.live_event_timeline_id_seq'::regclass);


--
-- TOC entry 5002 (class 2604 OID 25738)
-- Name: live_processing_logs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_processing_logs ALTER COLUMN id SET DEFAULT nextval('public.live_processing_logs_id_seq'::regclass);


--
-- TOC entry 5001 (class 2604 OID 25722)
-- Name: live_tracking_results id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_tracking_results ALTER COLUMN id SET DEFAULT nextval('public.live_tracking_results_id_seq'::regclass);


--
-- TOC entry 4996 (class 2604 OID 25667)
-- Name: live_video_uploads id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_video_uploads ALTER COLUMN id SET DEFAULT nextval('public.live_video_uploads_id_seq'::regclass);


--
-- TOC entry 4984 (class 2604 OID 25438)
-- Name: scenarios id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scenarios ALTER COLUMN id SET DEFAULT nextval('public.scenarios_id_seq'::regclass);


--
-- TOC entry 4986 (class 2604 OID 25451)
-- Name: timeline_events id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.timeline_events ALTER COLUMN id SET DEFAULT nextval('public.timeline_events_id_seq'::regclass);


--
-- TOC entry 5004 (class 2604 OID 34026)
-- Name: traffic_decisions id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_decisions ALTER COLUMN id SET DEFAULT nextval('public.traffic_decisions_id_seq'::regclass);


--
-- TOC entry 5009 (class 2604 OID 34044)
-- Name: traffic_demo_scenarios id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_demo_scenarios ALTER COLUMN id SET DEFAULT nextval('public.traffic_demo_scenarios_id_seq'::regclass);


--
-- TOC entry 5015 (class 2604 OID 34081)
-- Name: traffic_signal_monitoring id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_signal_monitoring ALTER COLUMN id SET DEFAULT nextval('public.traffic_signal_monitoring_id_seq'::regclass);


--
-- TOC entry 4982 (class 2604 OID 25423)
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- TOC entry 5320 (class 0 OID 34265)
-- Dependencies: 253
-- Data for Name: ambulance_operations; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ambulance_operations (id, ambulance_id, incident_id, current_latitude, current_longitude, destination_hospital_id, priority, eta_minutes, status, last_updated) FROM stdin;
1	AMB-101	611ae1cf-faf7-42b4-af55-5da3cc946fb8	19.017800	72.847800	365345f3-79bb-4391-ad3d-40ea263425da	CRITICAL	8	EN_ROUTE	2026-09-30 21:30:23.850295
2	AMB-104	32f052b2-a780-44d0-8e80-f6e05d8cdff9	19.039000	72.861900	e24d2ff5-5213-4756-89e0-ecb863c7a273	HIGH	14	EN_ROUTE	2026-09-30 21:30:23.850295
\.


--
-- TOC entry 5316 (class 0 OID 34216)
-- Dependencies: 249
-- Data for Name: hospital_agent_decisions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hospital_agent_decisions (id, hospital_id, decision_type, situation, decision, recommendation, expected_impact, confidence, status, created_at) FROM stdin;
\.


--
-- TOC entry 5314 (class 0 OID 34179)
-- Dependencies: 247
-- Data for Name: hospital_monitoring; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hospital_monitoring (id, hospital_id, total_beds, occupied_beds, available_beds, icu_total, icu_occupied, icu_available, emergency_capacity, emergency_occupancy, ambulance_queue, average_wait_minutes, patient_inflow, predicted_inflow, capacity_level, emergency_level, data_source, last_observed_at, updated_at) FROM stdin;
1	365345f3-79bb-4391-ad3d-40ea263425da	1000	850	150	120	115	5	50	48	3	25	Normal	Normal	HIGH	CRITICAL	SIMULATED_TEST_DATA	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
2	90935ac3-9bb5-42e8-92ad-454be5274eb2	300	210	90	50	30	20	25	10	0	5	Normal	Normal	MODERATE	NORMAL	SIMULATED_TEST_DATA	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
3	2f78e91c-7238-4028-869e-4a22592fa480	300	210	90	50	30	20	25	10	0	5	Normal	Normal	MODERATE	NORMAL	SIMULATED_TEST_DATA	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
4	e24d2ff5-5213-4756-89e0-ecb863c7a273	300	210	90	50	30	20	25	10	0	5	Normal	Normal	MODERATE	NORMAL	SIMULATED_TEST_DATA	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
5	e8a7d26e-7cfd-4127-8db5-3fa39c41535d	300	210	90	50	30	20	25	10	0	5	Normal	Normal	MODERATE	NORMAL	SIMULATED_TEST_DATA	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
6	184c6802-1ebf-498e-85bd-bac21ae7ab63	300	210	90	50	30	20	25	10	0	5	Normal	Normal	MODERATE	NORMAL	SIMULATED_TEST_DATA	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
7	6942a088-c122-4e64-8a63-611e6c2c0750	300	210	90	50	30	20	25	10	0	5	Normal	Normal	MODERATE	NORMAL	SIMULATED_TEST_DATA	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
8	d38c31f5-4555-4828-ae63-3c56ea287b4f	300	210	90	50	30	20	25	10	0	5	Normal	Normal	MODERATE	NORMAL	SIMULATED_TEST_DATA	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
9	0f0af16b-9a45-40b3-9aa8-75a51aa54505	1800	1720	80	120	115	5	100	95	6	45	HIGH	HIGH	CRITICAL	CRITICAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
10	37481af4-581f-4beb-9645-6a207032d4e9	1400	1250	150	100	85	15	80	70	3	25	HIGH	MODERATE	HIGH	HIGH	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
11	3a902613-79e2-4715-bc90-3d97f7fa5a5b	1300	1100	200	90	75	15	70	60	2	20	MODERATE	HIGH	HIGH	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
12	9f6d60c0-3191-45e5-afe5-c47f0fc0bcbc	1500	1350	150	110	95	15	90	80	4	30	HIGH	HIGH	HIGH	HIGH	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
13	af390dfe-0626-4dce-9ef6-363fcd900c14	800	600	200	60	45	15	60	40	1	15	MODERATE	MODERATE	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
14	108e2cee-481e-4dc7-bce3-79721bc9e7e8	400	320	80	50	40	10	40	25	0	10	NORMAL	MODERATE	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
15	2659b8a2-7c67-4cd1-b34b-1ea161ad36a5	350	250	100	45	30	15	35	15	0	5	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
16	fdb433fa-9804-4055-9d68-0fb51985de96	750	550	200	110	80	30	50	30	1	10	MODERATE	MODERATE	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
17	142afbdf-e8d4-45dc-81be-a3df12a5524c	350	240	110	50	30	20	45	20	0	8	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
18	2ae189f2-3950-4d42-b08d-012f4eea9891	350	210	140	45	25	20	30	10	0	5	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
19	0cfbf0d1-f7f9-4db1-a65e-5aff011f3c01	250	150	100	35	15	20	25	8	0	5	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
20	872b786c-078d-4e23-bea7-c06244c6bc9c	350	280	70	40	30	10	40	28	1	12	MODERATE	MODERATE	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
21	b39c9b83-6f28-48ff-907c-725d7ff31df9	240	180	60	40	25	15	35	22	1	10	MODERATE	NORMAL	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
22	b004021c-61e2-4671-89ed-f7802a7249ae	250	120	130	60	30	30	20	5	0	2	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
23	fb3997cb-c594-4691-8f6a-f57c555c23d7	600	580	20	80	78	2	30	28	3	40	HIGH	HIGH	CRITICAL	CRITICAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
24	c2c1ac72-3943-4ec6-9e87-3fb5ec0d5a8f	500	420	80	50	42	8	55	45	2	22	HIGH	HIGH	HIGH	HIGH	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
25	ebc446fe-f9fc-47ff-a524-43540d99ad13	300	210	90	30	20	10	45	30	1	15	MODERATE	MODERATE	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
26	ce15f155-aca1-470b-8a29-7ca59c3bbc64	300	260	40	30	25	5	40	35	2	20	HIGH	HIGH	HIGH	HIGH	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
27	b258f675-91bb-4994-adab-7c6f71caad41	320	240	80	40	30	10	50	35	1	18	MODERATE	MODERATE	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
28	6dfbf6ba-1fee-471a-b23f-4117d2631f8b	460	350	110	50	35	15	40	25	1	12	MODERATE	NORMAL	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
29	0fbb837c-4220-47ba-b8f9-e75c826c9939	520	300	220	60	30	30	35	15	0	8	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
30	1b12c61c-a649-426b-a9a3-fb85761eb996	300	220	80	45	30	15	40	25	0	10	MODERATE	MODERATE	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
31	c553a3ac-a164-4adf-9bd7-a6a1f5a0f28b	450	250	200	55	30	25	30	12	0	5	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
32	5d9bff13-3497-462a-895a-f2a84491e2b7	170	100	70	25	15	10	25	10	0	5	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
33	4077b0c3-31a6-4321-ba0b-6ccfb00291fd	268	190	78	38	25	13	35	20	1	10	MODERATE	MODERATE	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
34	651498ab-2a5a-4f88-bb2c-0643b0160b3a	250	140	110	37	20	17	30	12	0	5	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
35	f89f10d4-88e4-4ab6-87ba-e858b2897b33	210	120	90	30	15	15	25	8	0	5	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
36	3dfbbaa4-c0ec-4be7-aaff-1a1fa15c7e72	300	270	30	30	28	2	40	36	3	25	HIGH	CRITICAL	HIGH	CRITICAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
37	63f26211-4dfb-4e06-897d-ae1ec0b61f7a	120	70	50	15	8	7	20	5	0	2	NORMAL	NORMAL	NORMAL	NORMAL	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
38	1430acd7-bf69-4466-a501-38bb76ebf109	150	110	40	20	15	5	25	15	0	10	MODERATE	NORMAL	MODERATE	MODERATE	DEMONSTRATION_DATA	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
\.


--
-- TOC entry 5318 (class 0 OID 34239)
-- Dependencies: 251
-- Data for Name: hospital_resources; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hospital_resources (id, hospital_id, resource_type, total_capacity, in_use_qty, available_qty, threshold_percent, status, last_updated) FROM stdin;
1	365345f3-79bb-4391-ad3d-40ea263425da	Ventilators	50	45	5	85.0	HIGH	2026-09-09 00:13:32.387351
2	365345f3-79bb-4391-ad3d-40ea263425da	Oxygen (Cylinders)	1000	920	80	90.0	CRITICAL	2026-09-09 00:13:32.387351
3	365345f3-79bb-4391-ad3d-40ea263425da	Blood (Units - O+)	200	80	120	80.0	NORMAL	2026-09-09 00:13:32.387351
4	90935ac3-9bb5-42e8-92ad-454be5274eb2	Blood (Units - O+)	200	80	120	80.0	NORMAL	2026-09-09 00:13:32.387351
5	2f78e91c-7238-4028-869e-4a22592fa480	Blood (Units - O+)	200	80	120	80.0	NORMAL	2026-09-09 00:13:32.387351
6	e24d2ff5-5213-4756-89e0-ecb863c7a273	Blood (Units - O+)	200	80	120	80.0	NORMAL	2026-09-09 00:13:32.387351
7	e8a7d26e-7cfd-4127-8db5-3fa39c41535d	Blood (Units - O+)	200	80	120	80.0	NORMAL	2026-09-09 00:13:32.387351
8	184c6802-1ebf-498e-85bd-bac21ae7ab63	Blood (Units - O+)	200	80	120	80.0	NORMAL	2026-09-09 00:13:32.387351
9	6942a088-c122-4e64-8a63-611e6c2c0750	Blood (Units - O+)	200	80	120	80.0	NORMAL	2026-09-09 00:13:32.387351
10	d38c31f5-4555-4828-ae63-3c56ea287b4f	Blood (Units - O+)	200	80	120	80.0	NORMAL	2026-09-09 00:13:32.387351
11	90935ac3-9bb5-42e8-92ad-454be5274eb2	Ventilators	20	12	8	85.0	NORMAL	2026-09-09 00:13:32.387351
12	2f78e91c-7238-4028-869e-4a22592fa480	Ventilators	20	12	8	85.0	NORMAL	2026-09-09 00:13:32.387351
13	e24d2ff5-5213-4756-89e0-ecb863c7a273	Ventilators	20	12	8	85.0	NORMAL	2026-09-09 00:13:32.387351
14	e8a7d26e-7cfd-4127-8db5-3fa39c41535d	Ventilators	20	12	8	85.0	NORMAL	2026-09-09 00:13:32.387351
15	184c6802-1ebf-498e-85bd-bac21ae7ab63	Ventilators	20	12	8	85.0	NORMAL	2026-09-09 00:13:32.387351
16	6942a088-c122-4e64-8a63-611e6c2c0750	Ventilators	20	12	8	85.0	NORMAL	2026-09-09 00:13:32.387351
17	d38c31f5-4555-4828-ae63-3c56ea287b4f	Ventilators	20	12	8	85.0	NORMAL	2026-09-09 00:13:32.387351
\.


--
-- TOC entry 5312 (class 0 OID 34163)
-- Dependencies: 245
-- Data for Name: hospitals; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hospitals (hospital_id, name, area, latitude, longitude, hospital_type, emergency_available, status, created_at, updated_at) FROM stdin;
365345f3-79bb-4391-ad3d-40ea263425da	KEM Hospital	Parel	19.002500	72.842600	Public General	t	Active	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
90935ac3-9bb5-42e8-92ad-454be5274eb2	Lilavati Hospital	Bandra	19.050700	72.829100	Private Multi-specialty	t	Active	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
2f78e91c-7238-4028-869e-4a22592fa480	Tata Memorial Hospital	Parel	19.004800	72.842700	Specialty Cancer	t	Active	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
e24d2ff5-5213-4756-89e0-ecb863c7a273	Sion Hospital (LTMGH)	Sion	19.036300	72.861700	Public General	t	Active	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
e8a7d26e-7cfd-4127-8db5-3fa39c41535d	Hinduja Hospital	Mahim	19.034300	72.838400	Private Multi-specialty	t	Active	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
184c6802-1ebf-498e-85bd-bac21ae7ab63	Bombay Hospital	Marine Lines	18.940500	72.827300	Private Multi-specialty	t	Active	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
6942a088-c122-4e64-8a63-611e6c2c0750	Kokilaben Dhirubhai Ambani Hospital	Andheri	19.131100	72.826700	Private Multi-specialty	t	Active	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
d38c31f5-4555-4828-ae63-3c56ea287b4f	Nanavati Max Super Speciality Hospital	Vile Parle	19.096300	72.839600	Private Multi-specialty	t	Active	2026-09-08 23:57:51.161649	2026-09-08 23:57:51.161649
0f0af16b-9a45-40b3-9aa8-75a51aa54505	KEM Hospital	Parel	19.002500	72.842600	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
37481af4-581f-4beb-9645-6a207032d4e9	Sion Hospital (LTMGH)	Sion	19.039000	72.861900	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
3a902613-79e2-4715-bc90-3d97f7fa5a5b	BYL Nair Hospital	Mumbai Central	18.975000	72.822500	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
9f6d60c0-3191-45e5-afe5-c47f0fc0bcbc	JJ Hospital	Byculla	18.963300	72.833800	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
af390dfe-0626-4dce-9ef6-363fcd900c14	Cooper Hospital	Vile Parle	19.106500	72.836400	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
108e2cee-481e-4dc7-bce3-79721bc9e7e8	Hinduja Hospital	Mahim	19.033600	72.838400	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
2659b8a2-7c67-4cd1-b34b-1ea161ad36a5	Lilavati Hospital	Bandra	19.051000	72.827700	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
fdb433fa-9804-4055-9d68-0fb51985de96	Kokilaben Dhirubhai Ambani Hospital	Andheri	19.131400	72.824900	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
142afbdf-e8d4-45dc-81be-a3df12a5524c	Nanavati Max Hospital	Vile Parle	19.096300	72.839800	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
2ae189f2-3950-4d42-b08d-012f4eea9891	Jaslok Hospital	Pedder Road	18.971700	72.809000	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
0cfbf0d1-f7f9-4db1-a65e-5aff011f3c01	Breach Candy Hospital	Breach Candy	18.967000	72.805000	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
872b786c-078d-4e23-bea7-c06244c6bc9c	Wockhardt Hospital	Mumbai Central	18.975400	72.823600	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
b39c9b83-6f28-48ff-907c-725d7ff31df9	Hiranandani Hospital	Powai	19.122700	72.915700	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
b004021c-61e2-4671-89ed-f7802a7249ae	Asian Heart Institute	BKC	19.062000	72.864000	Specialty Cardiac	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
fb3997cb-c594-4691-8f6a-f57c555c23d7	Tata Memorial Hospital	Parel	19.004800	72.842700	Specialty Oncology	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
c2c1ac72-3943-4ec6-9e87-3fb5ec0d5a8f	Rajawadi Hospital	Ghatkopar	19.079200	72.898600	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
ebc446fe-f9fc-47ff-a524-43540d99ad13	Bhagwati Hospital	Borivali	19.245200	72.852400	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
ce15f155-aca1-470b-8a29-7ca59c3bbc64	Bhabha Hospital	Bandra	19.056000	72.832000	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
b258f675-91bb-4994-adab-7c6f71caad41	Shatabdi Hospital	Kandivali	19.201600	72.846100	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
6dfbf6ba-1fee-471a-b23f-4117d2631f8b	St. George Hospital	Fort	18.939200	72.838500	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
0fbb837c-4220-47ba-b8f9-e75c826c9939	GT Hospital	Fort	18.945500	72.832900	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
1b12c61c-a649-426b-a9a3-fb85761eb996	Fortis Hospital	Mulund	19.164300	72.943100	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
c553a3ac-a164-4adf-9bd7-a6a1f5a0f28b	Global Hospital	Parel	18.995000	72.840200	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
5d9bff13-3497-462a-895a-f2a84491e2b7	SL Raheja Hospital	Mahim	19.043500	72.843700	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
4077b0c3-31a6-4321-ba0b-6ccfb00291fd	Holy Family Hospital	Bandra	19.054300	72.831500	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
651498ab-2a5a-4f88-bb2c-0643b0160b3a	Saifee Hospital	Charni Road	18.952500	72.818300	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
f89f10d4-88e4-4ab6-87ba-e858b2897b33	Bhatia Hospital	Tardeo	18.964500	72.813100	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
3dfbbaa4-c0ec-4be7-aaff-1a1fa15c7e72	KB Bhabha Hospital	Kurla	19.066500	72.882500	Public General	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
63f26211-4dfb-4e06-897d-ae1ec0b61f7a	Zenith Hospital	Malad	19.186800	72.846500	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
1430acd7-bf69-4466-a501-38bb76ebf109	Apex Hospital	Borivali	19.229500	72.856500	Private Multi-specialty	t	Active	2026-09-21 18:14:07.65157	2026-09-21 18:14:07.65157
\.


--
-- TOC entry 5295 (class 0 OID 25635)
-- Dependencies: 228
-- Data for Name: incidents; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.incidents (incident_id, incident_type, severity, latitude, longitude, status, assigned_departments, timeline_events, created_at, upload_id) FROM stdin;
8a0d9e42-2e9e-4d0d-9e3b-1a6d5b8c1001	MEDICAL_EMERGENCY	High	19.0251000	72.8428000	Active	[{"role": "Patient transport", "status": "DISPATCHED", "department": "Ambulance"}, {"role": "Cardiac receiving", "status": "ALERTED", "department": "Hospital"}]	{"location": "Dadar West, Mumbai", "timeline": [{"time": "T+00", "event": "EMERGENCY_CALL_RECEIVED", "status": "COMPLETED"}, {"time": "T+02", "event": "AMBULANCE_DISPATCHED", "status": "COMPLETED"}, {"time": "T+04", "event": "PATIENT_ASSESSMENT_PENDING", "status": "ACTIVE"}], "confidence": 0.95, "description": "Suspected acute cardiac emergency", "scenario_id": "MEDICAL-CARDIAC-DADAR-001", "patient_count": 1, "response_status": "AMBULANCE_DISPATCHED", "ambulances_required": 1, "serious_patient_count": 0, "critical_patient_count": 1, "moderate_patient_count": 0, "estimated_hospital_eta_minutes": 10, "estimated_scene_arrival_minutes": 4}	2026-09-30 21:27:23.850295	\N
8a0d9e42-2e9e-4d0d-9e3b-1a6d5b8c1002	BUILDING_COLLAPSE	Critical	19.0326000	72.8481000	Active	[{"role": "Search and rescue", "status": "RESPONDING", "department": "Fire"}, {"role": "Perimeter control", "status": "DISPATCHED", "department": "Police"}, {"role": "Casualty evacuation", "status": "DISPATCHED", "department": "Ambulance"}, {"role": "Mass casualty receiving", "status": "ALERTED", "department": "Hospital"}]	{"location": "Matunga East, Mumbai", "timeline": [{"time": "T+00", "event": "COLLAPSE_REPORTED", "status": "COMPLETED"}, {"time": "T+02", "event": "FIRE_SERVICE_DISPATCHED", "status": "COMPLETED"}, {"time": "T+04", "event": "POLICE_PERIMETER_CONTROL", "status": "ACTIVE"}, {"time": "T+06", "event": "AMBULANCE_FLEET_MOBILIZATION", "status": "ACTIVE"}], "confidence": 0.93, "description": "Partial structural collapse with multiple reported casualties", "scenario_id": "DISASTER-COLLAPSE-MATUNGA-001", "patient_count": 8, "response_status": "MASS_CASUALTY_RESPONSE", "ambulances_required": 4, "serious_patient_count": 3, "critical_patient_count": 3, "moderate_patient_count": 2, "estimated_hospital_eta_minutes": 20, "estimated_scene_arrival_minutes": 9}	2026-09-30 21:28:23.850295	\N
8a0d9e42-2e9e-4d0d-9e3b-1a6d5b8c1003	INDUSTRIAL_ACCIDENT	High	19.0452000	72.8675000	Active	[{"role": "Hazard assessment", "status": "STANDBY", "department": "Fire"}, {"role": "Casualty transport", "status": "AVAILABLE", "department": "Ambulance"}, {"role": "Emergency receiving", "status": "ALERTED", "department": "Hospital"}]	{"location": "Sion Industrial Area", "timeline": [{"time": "T+00", "event": "ACCIDENT_REPORTED", "status": "COMPLETED"}, {"time": "T+02", "event": "INCIDENT_VALIDATED", "status": "COMPLETED"}, {"time": "T+04", "event": "AMBULANCE_ASSIGNMENT_PENDING", "status": "ACTIVE"}], "confidence": 0.91, "description": "Industrial machinery accident with worker injuries", "scenario_id": "INDUSTRIAL-SION-001", "patient_count": 4, "response_status": "RESPONSE_PENDING", "ambulances_required": 2, "serious_patient_count": 2, "critical_patient_count": 1, "moderate_patient_count": 1, "estimated_hospital_eta_minutes": 16, "estimated_scene_arrival_minutes": 7}	2026-09-30 21:29:23.850295	\N
dc8cf4a7-6b3d-4fb4-9327-7b234bebeda0	CRIME_ALERT	Critical	19.0180000	72.8436000	Active	\N	{"location": "Dadar Railway Station", "confidence": 0.98, "video_file": "7.mp4", "description": "Bank Robbery -> Suspect Vehicle Escape -> AI Tracking -> Pursuit", "scenario_id": "7"}	2026-09-30 22:44:56.715531	\N
\.


--
-- TOC entry 5299 (class 0 OID 25680)
-- Dependencies: 232
-- Data for Name: live_detection_results; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.live_detection_results (id, upload_id, frame, object_detected, confidence, bounding_box, tracking_id, timestamp_second) FROM stdin;
\.


--
-- TOC entry 5301 (class 0 OID 25699)
-- Dependencies: 234
-- Data for Name: live_event_timeline; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.live_event_timeline (id, upload_id, event_type, frame, timestamp_second, description) FROM stdin;
1	2	ACCIDENT	450	15	{"severity":"Critical","injuries":2,"roadBlocked":true}
\.


--
-- TOC entry 5305 (class 0 OID 25735)
-- Dependencies: 238
-- Data for Name: live_processing_logs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.live_processing_logs (id, upload_id, log_stage, message, logged_at) FROM stdin;
\.


--
-- TOC entry 5303 (class 0 OID 25719)
-- Dependencies: 236
-- Data for Name: live_tracking_results; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.live_tracking_results (id, upload_id, tracking_id, object_class, current_latitude, current_longitude, speed_kmh, timestamp_second) FROM stdin;
\.


--
-- TOC entry 5297 (class 0 OID 25664)
-- Dependencies: 230
-- Data for Name: live_video_uploads; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.live_video_uploads (id, filename, filepath, camera_id, location, status, uploaded_at, video_hash, duration_seconds, fps, processing_started_at, processing_completed_at) FROM stdin;
1	videoplayback.mp4	C:\\Users\\mange\\Documents\\React\\smart-city-platform\\backend\\src\\api\\public\\uploads\\1786459905423-983769999-videoplayback.mp4	CAM-101	Ruia College Road	Failed	2026-08-11 20:21:45.594029	b466b80af2e94f4f6c50d6e89df2f3db7c88fa38ebc091388113175fcd88357e	\N	\N	2026-08-11 20:21:45.594029	\N
2	maharashtra-india---march-03-2025-tragic-pmp4-15176819a.mp4	C:\\Users\\mange\\Documents\\React\\smart-city-platform\\backend\\src\\api\\public\\uploads\\1786459964385-353731239-maharashtra-india---march-03-2025-tragic-pmp4-15176819a.mp4	CAM-101	Ruia College Road	Completed	2026-08-11 20:22:44.546843	0c1b42c29b17913e033dfcb3fdc16d046a92030dfbe22fb1bea65507e17c590a	\N	\N	2026-08-11 20:22:44.546843	2026-08-11 20:22:51.877353
\.


--
-- TOC entry 5293 (class 0 OID 25603)
-- Dependencies: 226
-- Data for Name: map_entities; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.map_entities (entity_id, entity_type, name, latitude, longitude, metadata, layer_group, created_at) FROM stdin;
20d9d61a-f2b7-4af9-b677-bef7bc09ba93	hospital	Lilavati Hospital, Bandra	19.0510000	72.8277000	{"icuBeds": 45, "capacity": 314, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
ecd5f7aa-5f79-48c6-b5ae-05ec67119d0f	hospital	KEM Hospital, Parel	19.0028000	72.8415000	{"icuBeds": 120, "capacity": 1800, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
5a8d14c0-8f00-472b-95df-8e4d0717f16e	hospital	Nanavati Hospital, Vile Parle	19.0963000	72.8398000	{"icuBeds": 50, "capacity": 350, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
23d00902-73c3-46f2-b569-d35a3598f073	hospital	Hinduja Hospital, Mahim	19.0336000	72.8384000	{"icuBeds": 60, "capacity": 400, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
a58eea8c-3835-413b-b80e-bf5b66505c7b	hospital	Bombay Hospital, Marine Lines	18.9405000	72.8285000	{"icuBeds": 100, "capacity": 725, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
96d25961-1a8a-44a5-9c03-10cccb5dde89	hospital	Breach Candy Hospital	18.9670000	72.8050000	{"icuBeds": 35, "capacity": 212, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
df0b4293-ee27-4888-8c59-8452cd86894e	hospital	Jaslok Hospital, Pedder Road	18.9717000	72.8090000	{"icuBeds": 45, "capacity": 350, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
7a8f50cf-6a19-4cdb-8905-e22ebc64edcb	hospital	Sion Hospital (LTMG)	19.0360000	72.8617000	{"icuBeds": 150, "capacity": 1900, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
3e6ab956-9857-4cc0-a5f1-a6e9b319c334	hospital	Hiranandani Hospital, Powai	19.1227000	72.9157000	{"icuBeds": 40, "capacity": 240, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
df405917-c32b-48f3-aa2e-dbdc01dede42	hospital	Kokilaben Hospital, Andheri	19.1314000	72.8249000	{"icuBeds": 110, "capacity": 750, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
7540bd21-8c8a-483d-96e9-80a2491a4137	hospital	Tata Memorial Hospital	19.0048000	72.8427000	{"icuBeds": 80, "capacity": 600, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
f4bb5190-55f3-4f4b-aaca-21665f0a2eff	hospital	Fortis Hospital, Mulund	19.1643000	72.9431000	{"icuBeds": 45, "capacity": 300, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
d2d3e139-f9b6-477e-9fd1-602ada3fc153	hospital	Holy Family Hospital, Bandra	19.0543000	72.8315000	{"icuBeds": 38, "capacity": 268, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
7e0b8a07-5701-41c8-9239-090ba6a44792	hospital	Bhatia Hospital, Tardeo	18.9645000	72.8131000	{"icuBeds": 30, "capacity": 210, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
bb4a87e5-579e-4a5c-a924-ee27978734ec	hospital	Saifee Hospital, Charni Road	18.9525000	72.8183000	{"icuBeds": 37, "capacity": 250, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
907525e6-0c1e-4f79-af21-ebbb4f13b8c2	hospital	Cooper Hospital, Juhu	19.1065000	72.8364000	{"icuBeds": 75, "capacity": 636, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
d1eb95c1-ace8-440e-aabc-d6dc8ca87e0e	hospital	Global Hospital, Parel	18.9950000	72.8402000	{"icuBeds": 55, "capacity": 450, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
c8351a5f-faa4-46ba-ad46-822056b84d7a	hospital	Wockhardt Hospital, Mumbai Central	18.9754000	72.8236000	{"icuBeds": 40, "capacity": 350, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
6a96df4d-fc6c-433a-873a-e5d08a7bb4ae	hospital	SL Raheja Hospital, Mahim	19.0435000	72.8437000	{"icuBeds": 25, "capacity": 170, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
3acddd96-eabc-44fe-b75d-8e141d09569c	hospital	Zenith Hospital, Malad	19.1868000	72.8465000	{"icuBeds": 15, "capacity": 120, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
3ba9926e-b98f-487e-814f-ca415d13e4f7	hospital	Rajawadi Hospital, Ghatkopar	19.0792000	72.8986000	{"icuBeds": 50, "capacity": 500, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
d02b396e-a6d2-4013-b8b7-fd92580dec98	hospital	Bhagwati Hospital, Borivali	19.2452000	72.8524000	{"icuBeds": 30, "capacity": 300, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
2275c268-c0e2-4c3b-b63b-18cf2cc85151	hospital	Shatabdi Hospital, Kandivali	19.2016000	72.8461000	{"icuBeds": 40, "capacity": 320, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
98d45afd-3e1b-4e3d-95be-f7c2f7177c35	hospital	GT Hospital, Fort	18.9455000	72.8329000	{"icuBeds": 60, "capacity": 520, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
8727c084-8da3-4800-9a5d-19a463d84d15	hospital	St. George Hospital, Fort	18.9392000	72.8385000	{"icuBeds": 50, "capacity": 460, "emergencyStatus": "Normal"}	health	2026-07-21 22:15:53.602237
03918d94-e111-47f7-b1eb-de848b09ef9e	police_station	Colaba Police Station	18.9150000	72.8250000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
17596975-dcd3-4da0-ac71-e7026fadbd62	police_station	Marine Drive Police Station	18.9351000	72.8235000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
c2df51f5-0755-4af4-ab25-6aaf27f4b42c	police_station	Bandra Police Station	19.0560000	72.8320000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
f5afa27f-051b-40ec-bbf3-8dd3eab6f796	police_station	BKC Police Station	19.0620000	72.8640000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
805f7bd6-b783-403a-b96e-8a849153003b	police_station	Andheri Police Station	19.1170000	72.8465000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
4100617c-7b80-4f77-a09f-edf4ec06b7e1	police_station	Dharavi Police Station	19.0445000	72.8550000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
3776b1b7-489d-42e6-88c7-83d6bd3df8f6	police_station	Dadar Police Station	19.0195000	72.8425000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
c67a6c95-a285-4369-9ff9-5d5976ac02a8	police_station	Powai Police Station	19.1230000	72.9055000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
c2aceb15-0b65-4d8d-85b9-88b5de4807be	police_station	Juhu Police Station	19.1025000	72.8285000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
8c9cb3fa-c1e0-49fd-846e-c418cb3eb83f	police_station	Khar Police Station	19.0710000	72.8350000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
60e186a8-b91d-4821-8154-9b036f260481	police_station	Worli Police Station	19.0005000	72.8160000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
4ff71b8e-d8e4-42a3-a492-38b512809907	police_station	Ghatkopar Police Station	19.0860000	72.9085000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
d8af1db0-f493-4482-a870-2b812f2d57ab	police_station	Kurla Police Station	19.0665000	72.8825000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
67a48eab-da5f-44ca-88cb-eda60c7e23b7	police_station	Borivali Police Station	19.2295000	72.8565000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
b242ac61-4c92-4130-a277-45adc709b814	police_station	Chembur Police Station	19.0525000	72.9005000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
29232e13-3c50-4801-b153-7591daa98f44	police_station	Malad Police Station	19.1855000	72.8485000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
2704b089-b69b-4601-ac26-056c5ec7d09a	police_station	Goregaon Police Station	19.1630000	72.8490000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
2a1e21b6-871b-44c1-9129-497a902dd66a	police_station	Mulund Police Station	19.1725000	72.9560000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
94d3bb2e-0483-4bee-931c-a82537645e4e	police_station	Sakinaka Police Station	19.1005000	72.8885000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
988f466c-0a70-49fb-9297-83c3883a6f44	police_station	Tardeo Police Station	18.9735000	72.8140000	{"status": "Active", "patrolsAvailable": 5}	police	2026-07-21 22:15:53.602237
0745377a-5c41-4897-a780-d3980018d321	cctv	Cam 0	19.0139530	72.8497940	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
5bfbd384-225d-466e-a937-d5a402095f14	cctv	Cam 1	18.9843680	72.8548420	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
03f54a1d-dae0-406e-92bd-626d58427fe2	cctv	Cam 2	19.1144970	72.8284590	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
839517a0-8d31-4ed8-98c3-d492e3f91aa8	cctv	Cam 3	18.9937520	72.8533770	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
50e35391-3b53-472e-b797-9a0f4d235c0c	cctv	Cam 4	18.9993620	72.8382120	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
585f3574-3068-4f60-857a-0f9b18882e5c	cctv	Cam 5	19.0958750	72.8449770	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
f3daef32-028a-4db5-a7e6-4d70d8f3c78a	cctv	Cam 6	19.1712630	72.8735050	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
f6d2e58d-6587-43c1-9e1b-c20190aaa735	cctv	Cam 7	19.2253580	72.8360260	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
9e7d9ee0-8fe0-4a27-93e5-17369f509e5b	cctv	Cam 8	18.9636910	72.8153290	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
18ae6e7a-5cca-4655-a3ad-2dd549c266b2	cctv	Cam 9	19.2294460	72.8594760	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
707f8505-cb15-43a4-98f4-81e6932706d8	cctv	Cam 10	19.1132450	72.9093180	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
08b127b3-5515-4aec-b382-bdb886c9b50d	cctv	Cam 11	18.9904570	72.8498420	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
be67a298-06fc-438c-a8b4-d141a2f6f5af	cctv	Cam 12	19.0945780	72.8549420	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
ad7e8787-6444-4c77-a354-60771ab82010	cctv	Cam 13	19.0172620	72.8379850	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
f22cb02c-fc44-4e85-b639-8974fb100d02	cctv	Cam 14	18.9255440	72.8255770	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
7fe80f31-d96c-4e74-925d-1990eb357bcb	cctv	Cam 15	18.9962890	72.8444400	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
46fef7fb-c2ee-4743-a628-3eca4aa0437b	cctv	Cam 16	18.9240920	72.8181010	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
f37d85cd-33a4-4349-957b-6d2b72d36e63	cctv	Cam 17	18.9953530	72.8486250	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
cc4f60e5-00e1-4208-914b-b6dd2b135cdb	cctv	Cam 18	19.2233210	72.8597860	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
dfa421d3-a85b-4353-942b-9d3d0a6fad93	cctv	Cam 19	18.9278480	72.8160970	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
62b69c75-4a8d-4ca9-82c2-a7c92570e2bc	cctv	Cam 20	19.1374320	72.8358030	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
56db45a0-91c3-4598-b9a1-45a7e1187b95	cctv	Cam 21	18.9169850	72.8193590	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
e0e8cf0c-1997-4528-a993-efdfe9570183	cctv	Cam 22	19.1772640	72.8492330	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
e1599e4c-08f6-450d-b659-8f992942cfc2	cctv	Cam 23	19.0693970	72.8709510	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
6ab1d084-7040-43e3-8c76-85ccc6550ce1	cctv	Cam 24	19.2118420	72.8769390	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
1257a387-2c9c-4ca6-9bf3-9b47e6e039f5	cctv	Cam 25	19.0969940	72.8304460	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
2fc455b0-a440-43f2-b614-6acfb5e76040	cctv	Cam 26	18.9365170	72.8373870	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
b1c6b79b-d90a-4d21-854f-595cb2c2a42a	cctv	Cam 27	19.0975320	72.9129230	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
65591b25-49cc-4ce3-9169-c4c45a8428c1	cctv	Cam 28	19.0287870	72.8467250	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
5308ffd5-e539-46d1-8c80-5196ca3ac5ee	cctv	Cam 29	18.9351310	72.8355820	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
6179d1f3-5db0-4a3b-b311-dddb1f4e5240	cctv	Cam 30	18.9992280	72.8266890	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
f9ac094d-e115-4445-9384-a09bff8425c8	cctv	Cam 31	19.1924030	72.8339200	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
fd2aaa65-9311-4004-99f6-e681803006c4	cctv	Cam 32	19.0202940	72.8361620	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
c042fd0e-f283-4485-914b-ae33aca60b65	cctv	Cam 33	19.1823530	72.8329940	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
596243a6-7e21-47bb-9920-0f73fe84affd	cctv	Cam 34	18.9318500	72.8151540	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
e3852985-cfa8-4b7a-8c78-929b01c2c1b8	cctv	Cam 35	19.0086700	72.8428380	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
86d09009-3260-4679-89f6-79ab4c630c98	cctv	Cam 36	19.0212480	72.8490050	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
01b1b8d4-f363-4850-bad9-6204ed0e94b4	cctv	Cam 37	18.9361800	72.8343860	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
6ba01fcc-52a5-4af1-bf1f-6508a69331dd	cctv	Cam 38	19.1071580	72.8400680	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
dae2d4cf-5a94-48e4-9cc8-94a26e23db76	cctv	Cam 39	18.9175290	72.8182960	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
23893dc1-56bb-4620-bb46-cb2fc0a22cbf	cctv	Cam 40	18.9219390	72.8373170	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
bb6db7c2-79cb-4516-8729-24f01e866c99	cctv	Cam 41	18.9954040	72.8458710	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
b4d72ddd-fa52-4ef3-b09b-7c6c3d278582	cctv	Cam 42	18.9589190	72.8269320	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
12d464e5-d070-4103-b828-0de65450f541	cctv	Cam 43	19.0035760	72.8464030	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
7ffbd726-5e7c-4639-8b91-08b7eedd1b1b	cctv	Cam 44	19.1800810	72.8702120	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
940def64-ec99-4134-a1e0-7899e61de850	cctv	Cam 45	19.0015290	72.8279460	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
16b3c93e-ad71-467b-8858-0d797535d0b6	cctv	Cam 46	18.9194480	72.8307910	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
b7d9806e-7154-48ec-9143-5c8cc3320cb0	cctv	Cam 47	19.1067730	72.9287220	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
4fb1c580-e670-48a7-8072-59d4c60433a9	cctv	Cam 48	19.1215210	72.8450680	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
6b65800c-c9e8-49a3-a2ae-f6fa57df5942	cctv	Cam 49	19.1752610	72.8552970	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
27a267bc-b7f2-462c-94fa-e0c0a2d71397	cctv	Cam 50	19.0078630	72.8488750	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
99158e2b-c441-4a37-9c14-6248e0dd9ca8	cctv	Cam 51	18.9962250	72.8324810	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
52adc008-8dea-4e53-a3fd-a3140fc1d6c8	cctv	Cam 52	19.0492650	72.8484490	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
a21d49fa-0850-4832-a5dd-e31611b45666	cctv	Cam 53	19.2032310	72.8520310	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
5b399a81-b088-422d-b9d8-6fdb8e72f1c9	cctv	Cam 54	19.0057100	72.8492680	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
225b1c9a-6948-42ff-afb4-a97fe41d9d93	cctv	Cam 55	19.0972450	72.9244490	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
a4a233a4-46ca-4169-b148-d48089d357a3	cctv	Cam 56	19.1829550	72.8935460	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
5af56e04-6f5c-4162-9f9a-9ced2e20c67d	cctv	Cam 57	19.1899020	72.8981020	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
9290aed7-478f-42f4-8b7d-0addff55c417	cctv	Cam 58	19.1970480	72.9068070	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
bd762762-1851-4770-b828-f60825fc809d	cctv	Cam 59	19.0886100	72.8564150	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
092b29f4-2d9e-4e18-9dcc-a560605bf9d9	cctv	Cam 60	19.2244960	72.8518760	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
653ba245-3caf-4ff2-8afa-a38403dc9738	cctv	Cam 61	19.1590840	72.9110010	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
b5768315-a200-463b-bd05-3de1aaec2c03	cctv	Cam 62	19.0949030	72.8822370	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
cb682156-1ca4-4325-9383-c9b3dcbd94f8	cctv	Cam 63	19.1238680	72.8714600	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
b471306f-133c-4db8-a0be-3e28c7ec01c5	cctv	Cam 64	18.9391970	72.8225710	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
b8cd9d43-ad9c-4abd-a8c3-4007f1dd3556	cctv	Cam 65	18.9647060	72.8191980	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
9b1e3472-0aa8-4f06-9bcd-464459aae6eb	cctv	Cam 66	19.0132520	72.8281220	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
e79a5bc4-ee68-4c10-80f2-5038ece3cbf4	cctv	Cam 67	19.1372190	72.8756340	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
e1c0a6c6-75ec-4f2e-b082-367b694d6785	cctv	Cam 68	19.1923210	72.8278360	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
f224e4ba-afc8-4ab5-bfe9-34fa2ab3320a	cctv	Cam 69	19.1028640	72.9070560	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
9f767c78-230e-4dde-ab48-181c05bea842	cctv	Cam 70	18.9501290	72.8283270	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
31f2b255-54db-4b79-a7bd-7214be784fac	cctv	Cam 71	18.9564200	72.8158800	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
37933a24-0df4-40dc-b0b6-0f93a337eade	cctv	Cam 72	19.1923640	72.8558190	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
f4a45e64-8076-4cfd-94a3-fd0871db010e	cctv	Cam 73	19.0088910	72.8488610	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
f00efb8f-99c7-4c9b-83d9-db611e385159	cctv	Cam 74	19.0051420	72.8451500	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
3bb9118d-11f7-4b4c-9ac8-0d3976ec6f48	cctv	Cam 75	19.0711170	72.8269010	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
494a0c94-ad98-413c-903c-d11cb3c0253c	cctv	Cam 76	18.9324050	72.8206110	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
e7b0ac8d-fd23-434e-b185-3c596086ba66	cctv	Cam 77	19.1910240	72.8917690	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
b56d76ef-321d-44af-8185-79033747cc88	cctv	Cam 78	19.1647420	72.8800930	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
c79319ad-29e2-4c04-96ff-07ded017d9c6	cctv	Cam 79	18.9500040	72.8159880	{"status": "Live", "aiDetection": "Active"}	traffic	2026-07-21 22:15:53.602237
cd50917f-5d5b-4748-9a6f-4defff7b1b60	fire_station	Fire Dept 0	19.0260010	72.8534430	{"engines": 4}	fire	2026-07-21 22:15:53.602237
713ba01f-d103-4e0c-bed0-9f3f577e2a7a	fire_station	Fire Dept 1	19.0000780	72.8513780	{"engines": 4}	fire	2026-07-21 22:15:53.602237
78f2701c-39ca-4d6e-b0de-7c10a60fdaa7	fire_station	Fire Dept 2	18.9449320	72.8306610	{"engines": 4}	fire	2026-07-21 22:15:53.602237
59a69864-9f20-48c7-b561-e3ee6ec7ed2a	fire_station	Fire Dept 3	19.1738130	72.8890020	{"engines": 4}	fire	2026-07-21 22:15:53.602237
060853ba-ccfd-4695-9b69-c43571af1be1	fire_station	Fire Dept 4	19.1885330	72.9028090	{"engines": 4}	fire	2026-07-21 22:15:53.602237
ab6f2912-09cf-482d-a658-d86931b7abd1	fire_station	Fire Dept 5	19.0815200	72.8567340	{"engines": 4}	fire	2026-07-21 22:15:53.602237
1148fdde-a299-4fde-b6dd-23e90a6346c8	fire_station	Fire Dept 6	18.9433420	72.8232420	{"engines": 4}	fire	2026-07-21 22:15:53.602237
52231dfc-e85b-4d50-b0ed-eafcb8fb13b3	fire_station	Fire Dept 7	19.0445040	72.8308650	{"engines": 4}	fire	2026-07-21 22:15:53.602237
c5ec9c29-f62f-4353-8efa-bacd5f00911b	fire_station	Fire Dept 8	19.2179710	72.8409500	{"engines": 4}	fire	2026-07-21 22:15:53.602237
3c42535e-7185-4923-a999-e2ae4c33ad17	fire_station	Fire Dept 9	18.9827680	72.8524390	{"engines": 4}	fire	2026-07-21 22:15:53.602237
ebf93e6d-82bc-4ee4-9d22-462c13165ed1	fire_station	Fire Dept 10	19.1251840	72.8591370	{"engines": 4}	fire	2026-07-21 22:15:53.602237
04e3fd7b-50e7-47fb-a5ba-8f8f86f720c6	fire_station	Fire Dept 11	19.1509600	72.8357470	{"engines": 4}	fire	2026-07-21 22:15:53.602237
c4ce4e0e-bf37-4788-87ca-7351b03b4954	fire_station	Fire Dept 12	19.0253380	72.8479440	{"engines": 4}	fire	2026-07-21 22:15:53.602237
0b0e3791-318f-45e3-ac88-f2fdcc885fb9	fire_station	Fire Dept 13	18.9698260	72.8203220	{"engines": 4}	fire	2026-07-21 22:15:53.602237
1e6a72eb-0497-462b-968b-0fed2e36dae0	fire_station	Fire Dept 14	18.9894150	72.8402180	{"engines": 4}	fire	2026-07-21 22:15:53.602237
e8029666-0fe8-404d-9a5f-42d2a812ac95	street_light	Light Pin 0	19.0280410	72.8309240	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
d8033361-8ed2-4866-a32f-3997e52e57c8	street_light	Light Pin 1	19.1131610	72.8495170	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5c551566-52e6-4418-a553-8adddeca4d33	street_light	Light Pin 2	18.9309660	72.8332630	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
148445e0-1e62-4852-bc65-abf65327547f	street_light	Light Pin 3	18.9637040	72.8295460	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
289e3e4d-d0d7-4426-8aae-a8bc55e996d4	street_light	Light Pin 4	18.9963290	72.8367320	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
a35e6fd3-4c70-4b8e-98e6-f26abd5bbd2f	street_light	Light Pin 5	19.0241350	72.8251880	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
834ea9ca-7fd0-483b-9cb6-78404165fa89	street_light	Light Pin 6	18.9920880	72.8374210	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
d2c2666f-4924-437b-a412-77a7c8187d2f	street_light	Light Pin 7	19.1229090	72.8838660	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5e260553-abb6-49ef-9af7-8b827ed927fc	street_light	Light Pin 8	18.9568740	72.8186870	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
6d22a927-17b9-4a9f-9cc3-1444137e7dbf	street_light	Light Pin 9	19.1498000	72.8586310	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
3efbf2db-6b19-40fd-a340-0f157b1e0d5e	street_light	Light Pin 10	18.9246940	72.8321810	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
3c2aa0fc-e85a-4a6f-bc46-86b81f2eee17	street_light	Light Pin 11	18.9433450	72.8242190	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
4f137e6c-ff9a-4f31-8327-f7d6b45f303d	street_light	Light Pin 12	18.9222810	72.8331970	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
8d2d3114-c0e0-4d15-a4c2-255de3f79299	street_light	Light Pin 13	18.9225330	72.8221270	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
af23f18d-8578-4a64-8d6e-90c9689960c2	street_light	Light Pin 14	18.9941530	72.8354300	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
058d1789-ec64-40ae-80a4-101d2566340b	street_light	Light Pin 15	19.0837650	72.9083050	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
27f79da5-6489-447d-8cd3-d7ba00b500a4	street_light	Light Pin 16	19.1665450	72.8255410	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
f747070a-dd61-4259-85e6-c6724f59b7f8	street_light	Light Pin 17	19.0483990	72.8454740	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
61b5d435-1a6b-4194-b28a-9c5123f9b0a6	street_light	Light Pin 18	19.1245980	72.8904760	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
e071d69c-af8b-4381-9c97-643fa3493888	street_light	Light Pin 19	19.0080770	72.8438660	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5c11910e-2cf9-4d52-aa69-aa2ffa1e3372	street_light	Light Pin 20	19.0015980	72.8263190	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
79a22a21-9036-4996-b693-fd1c363ab74a	street_light	Light Pin 21	18.9209690	72.8252530	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
35983c14-e5df-4cc6-92a2-899f0b7f254c	street_light	Light Pin 22	19.0618760	72.8734960	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
c1a6d703-69ed-4f93-8f03-96959606e757	street_light	Light Pin 23	18.9453220	72.8343690	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
171875a4-8b80-44ae-8bb3-c460ba9768fa	street_light	Light Pin 24	19.0671380	72.8747560	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
4c9a1617-0e1d-4fc1-99b7-61af474a0968	street_light	Light Pin 25	19.2143240	72.9213400	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
337df25f-5d53-45eb-9449-d68931f036ae	street_light	Light Pin 26	19.0282940	72.8314570	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
dbaf97ad-757e-4dd2-a783-2bb3174e385a	street_light	Light Pin 27	18.9480150	72.8350710	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
7b0467d2-afec-4858-9e32-ff20ed2ea208	street_light	Light Pin 28	18.9578510	72.8211790	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
c71b5c7e-2cff-4abf-8f55-6ca2e8dae454	street_light	Light Pin 29	18.9676980	72.8355560	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
156837d3-56c4-4e39-aad6-243ca17f27a7	street_light	Light Pin 30	19.1182000	72.8407870	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5298942c-bab8-432a-bd0e-566087a8ead5	street_light	Light Pin 31	18.9647980	72.8372050	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
1e79492f-93ed-426c-98c7-50a8bffd0b69	street_light	Light Pin 32	19.1048550	72.9037810	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
584195fc-73e3-494f-82c0-7cde16b8e7e8	street_light	Light Pin 33	18.9581270	72.8294130	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
aae5bc25-26bb-43d1-8386-46b84916c55c	street_light	Light Pin 34	18.9592350	72.8361600	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2f5dd484-f73f-4a42-b4a9-7f5630689c61	street_light	Light Pin 35	19.0198190	72.8322970	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
71345282-3948-4004-9c13-36b4ac71c3e2	street_light	Light Pin 36	19.0220620	72.8447480	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ea007431-1bb2-4975-b767-f3dbff0e3c9d	street_light	Light Pin 37	18.9586570	72.8346510	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
278f532e-aeef-4806-88f1-df1c44efe013	street_light	Light Pin 38	19.1336600	72.8705910	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
fa64b30e-5fa8-4d3e-b80f-c18e1be282bf	street_light	Light Pin 39	18.9670810	72.8226290	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
19e46422-f024-42d9-b9e8-0095766bdf0f	street_light	Light Pin 40	19.0408770	72.8481150	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
0262c841-2978-48fb-b618-cb66bfbc1377	street_light	Light Pin 41	19.1659300	72.8343350	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
4d61a507-6c82-41c4-af40-44d7fc3b990f	street_light	Light Pin 42	19.0035750	72.8336560	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
f31f6d3c-99c2-43f6-9d6f-45b2394709aa	street_light	Light Pin 43	19.0985550	72.8397870	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
69ca6b55-db16-4689-a08d-3cabe308e547	street_light	Light Pin 44	19.1273010	72.8859900	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
4a8e9927-2494-4a7f-b416-5030982ab776	street_light	Light Pin 45	19.1399850	72.8380470	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
630bc439-7ec9-45f6-ab26-6c0e98f2c413	street_light	Light Pin 46	18.9369150	72.8270010	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
efc29765-06bc-4ca2-930c-b34193739e4e	street_light	Light Pin 47	19.2276560	72.8567800	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
57e3587f-7977-48c2-905b-b35d09949851	street_light	Light Pin 48	19.0892840	72.9062610	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
83f5b822-1a43-4932-97c3-29c5023b5637	street_light	Light Pin 49	19.2138360	72.8381700	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
19219b18-48e2-498f-9f84-3348119ff122	street_light	Light Pin 50	19.1877960	72.9288100	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
7ec94845-d738-47ec-b57b-4520f366f6fd	street_light	Light Pin 51	19.1238860	72.8988200	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
52502139-1474-4f0c-87ee-94ad07f3be6b	street_light	Light Pin 52	18.9810860	72.8511600	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
4f755bb3-702b-40b3-8880-874fe8a54971	street_light	Light Pin 53	19.0159860	72.8305090	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
72733e81-091c-4b95-aded-0edb0eb06a81	street_light	Light Pin 54	19.0888170	72.8884650	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
c2141efd-e521-4411-a3af-7e6126eeac9a	street_light	Light Pin 55	18.9572170	72.8304350	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
82b9d64b-fb2d-46cc-837a-4f53158ee629	street_light	Light Pin 56	19.1248390	72.8542880	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
17cd712b-e2e2-4322-98d0-e159a3402ad5	street_light	Light Pin 57	18.9469590	72.8295380	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ed4f75b6-b15f-40fa-8e1b-3609ffa5e8fe	street_light	Light Pin 58	19.1862520	72.9209700	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
dcda1844-6bee-4647-99c8-b2d4b6d6b4b7	street_light	Light Pin 59	19.1477460	72.8580110	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
6c85ed25-19ac-4fcb-b1ff-87e5b9e4f2cb	street_light	Light Pin 60	18.9663850	72.8194720	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
6f9600e6-687c-4d3a-bfb6-5035df19898c	street_light	Light Pin 61	18.9256140	72.8247380	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
69c05153-2b9d-4bc7-8d86-90e61f3adf29	street_light	Light Pin 62	18.9942840	72.8527570	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
d21a41a8-82ca-4af5-bb1f-11be7abf0bbe	street_light	Light Pin 63	18.9922940	72.8356380	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5dae550d-f910-40b0-a69c-d46f43bc3e4d	street_light	Light Pin 64	19.1105450	72.8425470	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
f02e47d1-4d76-444a-9e36-965d446924e3	street_light	Light Pin 65	19.1551570	72.9259200	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
91661ad3-e979-4f2a-b06c-1a847cde4b2a	street_light	Light Pin 66	18.9660750	72.8157910	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
36150247-6a6d-439b-9177-bc63aab7b936	street_light	Light Pin 67	18.9954490	72.8373140	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
3cbf2abe-a1e6-4e2f-873e-ea0735bdf136	street_light	Light Pin 68	18.9837530	72.8381630	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
071fb6af-23f8-4acd-a772-6961d06daa29	street_light	Light Pin 69	18.9905480	72.8378780	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b213a29a-ff81-40e7-8d3c-045a08ac2d84	street_light	Light Pin 70	19.0887700	72.9162350	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
6d452aaf-171b-4a0b-8146-c7b2831ef9be	street_light	Light Pin 71	18.9932820	72.8443680	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
afca8f89-27ec-4338-9bdf-3693cd010346	street_light	Light Pin 72	18.9419980	72.8202320	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
4cf3dd9e-b697-410a-9237-8ae6d4f43261	street_light	Light Pin 73	18.9914100	72.8354170	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
8e3cbda9-3741-4380-91aa-6df24186fe72	street_light	Light Pin 74	19.2278660	72.8541820	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
754d707a-e3c0-469b-91df-27d0d85676a9	street_light	Light Pin 75	18.9209650	72.8270430	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b581db68-d0e9-43db-97cc-b2f47493c149	street_light	Light Pin 76	19.0091220	72.8496930	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
de949d45-2d95-40e0-9e53-65a3bb1022a3	street_light	Light Pin 77	19.1455430	72.8975080	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
69955a88-af5a-41d1-afb3-1f7f494f2d3c	street_light	Light Pin 78	19.2162030	72.8305630	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
94a85ab4-0c4a-460b-8f82-14704d230e95	street_light	Light Pin 79	18.9644760	72.8269770	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
680913ed-2605-4e7f-af45-c739019d6c01	street_light	Light Pin 80	19.1049270	72.8495630	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
dc8810eb-9280-41b9-ba36-92e9f11dc6fb	street_light	Light Pin 81	19.2117650	72.9267930	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
8cf3c394-5fe9-4155-962c-b73dae7ea084	street_light	Light Pin 82	19.2152600	72.9128610	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
7f023053-46f0-420d-80fe-f1b96e7f9ece	street_light	Light Pin 83	19.2123700	72.8869540	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
1c9d6447-4958-4d08-9308-abf23950fe98	street_light	Light Pin 84	19.2129790	72.8792200	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5dda4d2b-41bf-491e-9b4a-c52e237153d6	street_light	Light Pin 85	19.1638590	72.8296390	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
d7c8c6c2-2fd8-400c-a2a8-4533c76827d3	street_light	Light Pin 86	19.1260720	72.8520470	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
54d04a56-1d05-44a8-ae59-596a7405c28d	street_light	Light Pin 87	18.9606580	72.8153960	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
c737d7f0-b706-49d5-8ff6-337619048b43	street_light	Light Pin 88	18.9538960	72.8360140	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2aa6b925-2e58-4ae9-b895-6714cbcf6c30	street_light	Light Pin 89	19.0941680	72.8777340	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ce35c24d-275f-443b-96a4-c6aa17128c89	street_light	Light Pin 90	18.9365170	72.8358130	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
a1f96244-b8a4-4d18-8ee3-90f9ce4fd83f	street_light	Light Pin 91	19.2063560	72.8377080	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
179ae499-0e3e-4f13-a76d-fec614c9b84b	street_light	Light Pin 92	19.1570450	72.8931980	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2b9211be-66be-4181-b6c9-817d27b5fa12	street_light	Light Pin 93	19.0845090	72.8281620	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5025ff48-036a-4aa8-95b8-1b683d09b1d4	street_light	Light Pin 94	18.9485040	72.8188980	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5d912da7-a1b7-419c-a708-0dada325e3a8	street_light	Light Pin 95	19.0627490	72.8331320	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
1e07c7c5-221b-47b5-b315-daa2611912c1	street_light	Light Pin 96	18.9489490	72.8269210	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
18dc4fe5-3f26-46c0-8449-3a7c1808fbc9	street_light	Light Pin 97	19.0093760	72.8371070	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
f9a779d3-cf86-4f50-a05f-8c984efce2b5	street_light	Light Pin 98	19.1133120	72.8375960	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
3b621d2e-fadc-414a-893b-677dbe39f7e3	street_light	Light Pin 99	19.1322920	72.8898690	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
a007a3b2-0938-43bb-a614-0b03ac6fc6dc	street_light	Light Pin 100	19.0022290	72.8309170	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
9c70a724-3663-4503-bc03-42c9d8199735	street_light	Light Pin 101	19.1406140	72.8433370	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
3d161bc3-5e19-4b23-86cd-2938e6fea58b	street_light	Light Pin 102	19.0656950	72.8409720	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
86a10f89-9cf5-4ebc-b82e-c951ae608a12	street_light	Light Pin 103	19.0257530	72.8432770	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
76f26c31-56c6-4c5d-9fa1-e538ca54df49	street_light	Light Pin 104	18.9635650	72.8179730	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5ac8d80a-91ce-4d49-989a-e23ac026f7fd	street_light	Light Pin 105	19.0111230	72.8526510	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
30b72d9c-e934-4694-a559-3f4a05192eac	street_light	Light Pin 106	19.1429760	72.8429480	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
aa8d15af-163a-404d-a3ba-3439bfebbb87	street_light	Light Pin 107	19.1935890	72.8305440	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
64f6f495-222d-4202-8e70-8fd7c5c3535a	street_light	Light Pin 108	18.9285430	72.8346170	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
e8c6d96a-39d6-47f7-804b-ab8573411d73	street_light	Light Pin 109	19.1046970	72.8408370	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2ca2f3b8-2701-48db-91dd-a6acb2931da2	street_light	Light Pin 110	19.1825260	72.8407530	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
02971f4f-3dff-47a0-a923-ebc091e3f26a	street_light	Light Pin 111	18.9492060	72.8164040	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5dd40730-3591-49cb-985c-420a08e70c36	street_light	Light Pin 112	19.0175820	72.8494630	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b931962d-e7f4-43d9-a587-532c95087a2a	street_light	Light Pin 113	19.1451700	72.8793910	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
441a7ad7-0570-42e9-bca5-c3eaa38f1d9b	street_light	Light Pin 114	19.1726820	72.9081610	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
01fb4152-a853-4b1e-bab8-5b5ea2b043c8	street_light	Light Pin 115	19.1451260	72.8754820	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
1b2c16a2-68ba-4745-9c09-d409d09421c2	street_light	Light Pin 116	18.9944180	72.8515190	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
afcc99f4-6cfe-4d52-860a-eb14b4285931	street_light	Light Pin 117	18.9805420	72.8385320	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
cca80656-3894-4e13-99a8-c3e388fd4ffd	street_light	Light Pin 118	19.0632900	72.8385820	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
e0c155c9-a1cd-49de-8dc2-f5f61e1ac91c	street_light	Light Pin 119	19.1893490	72.8879950	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b230a08a-f771-4aaf-b9ad-3898c3db14c4	street_light	Light Pin 120	19.2000050	72.8315560	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
6ccaa109-3d13-4e6c-89a5-b761e948dbfd	street_light	Light Pin 121	18.9359240	72.8227750	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
61581027-da29-4829-a056-e556387158e8	street_light	Light Pin 122	19.1974010	72.8418480	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
4b853740-5c91-41c4-b58f-3ace589705dc	street_light	Light Pin 123	19.1662380	72.8708070	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
532aaf66-bf11-48e6-828d-f798c5e4244b	street_light	Light Pin 124	19.0070580	72.8518210	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ef113608-ca46-4f97-bfa4-27f269e3c252	street_light	Light Pin 125	19.1123510	72.9012260	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
a5b962b4-7d98-48af-8701-a561a5e91229	street_light	Light Pin 126	19.0702680	72.8988790	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
0e23b33d-f6b5-44ff-9d8b-d10f40d15f75	street_light	Light Pin 127	19.0945950	72.8795640	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5ab6116c-3e98-421f-92bf-583547a486cc	street_light	Light Pin 128	19.0134490	72.8520550	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ac05b0f1-7edb-42dd-9451-abb85fe30dac	street_light	Light Pin 129	19.0508860	72.8400410	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
fbc9e5eb-cd91-4a04-ba0b-000b04454c03	street_light	Light Pin 130	18.9587200	72.8288010	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
48a026a9-10eb-4566-929d-306fd9306e3d	street_light	Light Pin 131	19.2085320	72.8867130	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
1ea4697b-262c-4772-bc68-9e2e33cbc803	street_light	Light Pin 132	19.1370930	72.8719070	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b2c342a0-42d1-4eca-921c-064a995dbd61	street_light	Light Pin 133	19.0932360	72.8531240	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
17f621ca-e9b0-4c7f-8e50-a9d57709f951	street_light	Light Pin 134	18.9290900	72.8166840	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
d35f5d6f-6bbe-4540-8dd6-ed009b175adc	street_light	Light Pin 135	18.9299500	72.8358420	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
f5e53964-7a4f-4013-a58b-6af74fd00ac7	street_light	Light Pin 136	19.1714050	72.8868760	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
e00c8de4-7a85-4313-8078-5ab70874e342	street_light	Light Pin 137	18.9853240	72.8414950	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ee418f55-fee0-44a6-b937-33612f85bae7	street_light	Light Pin 138	19.2115410	72.9019170	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
03266747-4678-4b47-bbb1-e1752060c191	street_light	Light Pin 139	18.9572020	72.8150820	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
a5261d65-5cba-48a9-88a8-b4ca92344cbf	street_light	Light Pin 140	19.2023770	72.9057390	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
a8f315e3-cd6c-4a5e-aa41-99f8c5a74cf5	street_light	Light Pin 141	19.1744920	72.9194420	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
bd50c30e-8f15-435a-b5ac-c0aad548cdaf	street_light	Light Pin 142	19.2186260	72.9278690	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
da35d3f3-345a-4c2e-a30f-d4c8a78c5c4d	street_light	Light Pin 143	19.1013620	72.8996030	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5aabaca9-43bf-4729-b841-9918c9dea6ae	street_light	Light Pin 144	19.0251320	72.8433500	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
81bdcd9b-f9c9-40fc-b257-89a8b3230ff6	street_light	Light Pin 145	19.1183620	72.8403190	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
0726a3e1-4535-4813-b477-b11444e52867	street_light	Light Pin 146	19.2181370	72.9080800	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2e8ce56d-3967-45c1-9541-efc2c964dad9	street_light	Light Pin 147	18.9462770	72.8234330	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
8ae9890c-7751-4775-a64e-5c02c0262cf5	street_light	Light Pin 148	19.0587980	72.8368880	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
52a130aa-291e-4776-8b68-90afe9339ca4	street_light	Light Pin 149	18.9537760	72.8231760	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
92594763-8f03-4b7b-bf02-4c69a96eb0fa	street_light	Light Pin 150	19.1938340	72.8871090	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
0f4a04e4-99c6-44a7-8674-47a18807e1d5	street_light	Light Pin 151	18.9843950	72.8471130	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
e145cac0-0036-43d4-9d22-07ab0d5e5c1d	street_light	Light Pin 152	18.9438180	72.8322720	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
d98eadc2-16b2-4031-9079-943d2e3a57dd	street_light	Light Pin 153	18.9413340	72.8301240	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
16328b19-58bf-4321-a476-dac6fb5582c0	street_light	Light Pin 154	18.9561100	72.8160540	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
f18c6c76-1c92-40a5-83d7-10894de0ae90	street_light	Light Pin 155	19.0277900	72.8425080	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
58ba58e7-82d6-4c14-be8b-61b2f013d40a	street_light	Light Pin 156	19.1366570	72.9036470	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
8bb91c94-735b-45a7-b75d-4e25a02d1dcf	street_light	Light Pin 157	19.0175200	72.8277820	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
7c06fe6c-2b8c-4c9c-becc-5edccad8c591	street_light	Light Pin 158	18.9430000	72.8369080	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
e09608ed-1eb2-40b3-9028-cf2bf5839275	street_light	Light Pin 159	19.1000670	72.8584080	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
84ae5ca1-596a-4047-941b-413239a1c789	street_light	Light Pin 160	19.0012160	72.8511440	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
61d77539-dd76-482e-96fd-a8e17fdb4f58	street_light	Light Pin 161	19.1344790	72.9284210	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
22ea1466-9b6f-4b13-9dcf-183c1d05a68d	street_light	Light Pin 162	19.1603980	72.8751760	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b8f51583-0d8f-4284-9c1a-56f55dc0a164	street_light	Light Pin 163	19.0242840	72.8490250	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b7f58fac-c2d3-47f5-a225-dfc38755755d	street_light	Light Pin 164	19.1746070	72.9084040	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
75897fa2-c3c0-42d1-bb3c-2057f963ff17	street_light	Light Pin 165	19.0244200	72.8294900	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
1dbb5718-b006-4394-a613-1d5897c41d6c	street_light	Light Pin 166	19.0155940	72.8396970	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
be2b5ae2-fbe9-4233-9926-cd9a4ba59768	street_light	Light Pin 167	18.9555250	72.8171150	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
a4d74392-340e-443c-a68c-a7901a8800d3	street_light	Light Pin 168	18.9545050	72.8344880	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
7712830a-064b-4224-8c7c-ef308d1ba182	street_light	Light Pin 169	18.9527990	72.8315850	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
cd9b30e7-2518-4316-9d3c-40c5e1e1b287	street_light	Light Pin 170	19.0819290	72.8988190	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b1b05e88-e5f0-48de-835a-cefcfd4070db	street_light	Light Pin 171	19.1032250	72.9083470	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
236c76f0-398c-46ac-b3bd-45f7d27bf654	street_light	Light Pin 172	18.9376350	72.8218020	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
91320b77-80e8-4106-920e-4d6348084f40	street_light	Light Pin 173	18.9233630	72.8367530	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ad430cbf-2e83-4b9d-99fa-49a25f89092f	street_light	Light Pin 174	18.9989230	72.8522410	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
c91b46c5-2f92-499f-808e-5c479e415c27	street_light	Light Pin 175	18.9375460	72.8338100	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
4c3276bf-1a53-45ad-ba25-0b7f9ef98b34	street_light	Light Pin 176	18.9645540	72.8186500	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
d491033f-8674-40be-8cb7-dd3e091224b5	street_light	Light Pin 177	19.0614310	72.8906300	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
36200d04-26dc-4ab9-bc59-be70115f7fe8	street_light	Light Pin 178	18.9429080	72.8332770	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
27a80905-08c4-43ed-93c9-2bb486e6679d	street_light	Light Pin 179	18.9415240	72.8225350	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
fe58b7ab-d28e-4dd0-9383-8e2373f27271	street_light	Light Pin 180	19.1200810	72.8353760	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
9e915a97-5085-4193-a8ed-d5b1ae18fcf6	street_light	Light Pin 181	18.9346180	72.8239920	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b1a18cab-6758-429a-8e8f-acb387772e02	street_light	Light Pin 182	19.1800140	72.9296430	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
fb0ded23-3eee-412e-b6bb-cb336392d60a	street_light	Light Pin 183	19.0120030	72.8369330	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
14a82d5f-89a5-426e-a26e-5e0ee8b9289e	street_light	Light Pin 184	19.1891950	72.8303820	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
d664bc36-b886-48af-9316-c845b77f3cda	street_light	Light Pin 185	19.1142930	72.8305680	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
35369b7f-6489-4db5-a918-f3e2707fc044	street_light	Light Pin 186	18.9966900	72.8441950	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
13790b14-4e6f-49a2-b251-71ed2eb9c2f8	street_light	Light Pin 187	19.2130650	72.8302640	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
39327555-79ce-4c8f-9da3-258bb119fc96	street_light	Light Pin 188	18.9581780	72.8333490	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
12c5fbb9-7a95-4718-9539-e46fa58b9a05	street_light	Light Pin 189	18.9963280	72.8490580	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
baf084cd-c02c-4d19-8dd5-76febe018c12	street_light	Light Pin 190	19.0701030	72.9118440	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
43beb869-8d8e-4155-ae15-db1bd5e876ba	street_light	Light Pin 191	19.1992130	72.8578260	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
c673b5a1-d02b-4d2d-bfb4-b2c31bde6d5f	street_light	Light Pin 192	19.0240870	72.8425290	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
33c12a1f-992b-44fb-b4d5-02be73253a9d	street_light	Light Pin 193	18.9528010	72.8248610	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b8a694a8-e705-4083-a819-94f40b2871bc	street_light	Light Pin 194	19.1627210	72.8866480	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
80f7b0cb-658d-4dc7-9c6a-1aa79eb7b977	street_light	Light Pin 195	18.9465720	72.8220690	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
afc00a55-5140-4c46-af48-0c3656827598	street_light	Light Pin 196	18.9547170	72.8243830	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
99d6d511-b7f6-4e01-9431-7d20ee0cd27b	street_light	Light Pin 197	19.0020220	72.8484710	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
91d7413b-31e8-45a7-a663-a2af381391c2	street_light	Light Pin 198	19.0562360	72.8420530	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
053517df-e685-46cd-9508-7239c85057dd	street_light	Light Pin 199	18.9167520	72.8236310	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
c665fc57-4802-4e42-8247-22947261e33a	street_light	Light Pin 200	19.1132390	72.9167790	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
87e91788-7f6c-4efd-9704-9bb527bc11b2	street_light	Light Pin 201	19.2249920	72.8528660	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
90d87d68-76e8-4abe-8a9c-4f6a495a8a74	street_light	Light Pin 202	19.0101780	72.8512790	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
74a038b7-496b-4fc7-a415-74561de3081d	street_light	Light Pin 203	18.9866640	72.8459730	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
e3902c91-8154-46e6-8481-fb3fcf6031da	street_light	Light Pin 204	18.9639910	72.8368440	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
1fcf4714-6a07-4b4c-8d62-1c6194936dbb	street_light	Light Pin 205	18.9670760	72.8180950	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
e68a08fa-4ad9-42cf-8e0f-b879c644bfe0	street_light	Light Pin 206	19.1661410	72.9260370	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
af4d9723-26df-4f5a-9c38-135afa924267	street_light	Light Pin 207	18.9907560	72.8284830	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
fee05356-225b-4850-9714-8351e7255e09	street_light	Light Pin 208	19.1878380	72.8504450	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5d8fb968-580a-4c7d-ac62-53cc43653413	street_light	Light Pin 209	18.9384800	72.8275540	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
09a0e8dd-62fe-494e-81de-9da29e1f52d9	street_light	Light Pin 210	18.9364750	72.8352540	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
e5ce8ffd-383e-49ab-9d0a-160edfca0883	street_light	Light Pin 211	19.0840370	72.8537970	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
c6b5df15-7cda-46e2-8dd6-0e36a3beff59	street_light	Light Pin 212	18.9339110	72.8204320	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
cfc88200-8e8e-49fa-bf24-0baef16bd590	street_light	Light Pin 213	19.1262150	72.8532500	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
c6133dbb-f4cd-4437-acd5-2c7941a8a7a3	street_light	Light Pin 214	19.1489080	72.8840830	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2c155f64-68ae-40af-9e56-2479fe48cc38	street_light	Light Pin 215	18.9375040	72.8276930	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
6ed03894-4f3a-441f-9d90-cc5334f5d6d3	street_light	Light Pin 216	18.9219290	72.8182750	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
65451b2e-a89a-4db9-beda-9c529f2ac9d3	street_light	Light Pin 217	19.1015170	72.8928300	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
b15f6feb-ba4b-4469-a9a9-cc874c49717a	street_light	Light Pin 218	19.0889640	72.9194490	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
24480805-ec02-42de-925c-b668642b6219	street_light	Light Pin 219	19.0206210	72.8365980	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
e437fc11-574c-44f4-9a50-00752b2a70f6	street_light	Light Pin 220	19.1860090	72.8585940	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
9036be1c-3efe-4ef2-85a0-1f2d62f11218	street_light	Light Pin 221	19.0054880	72.8436760	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ae8021f8-a691-49fd-a3be-769d8c536e07	street_light	Light Pin 222	19.0780990	72.8300690	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2aac77ba-f7ac-454d-8b35-09188635d6d1	street_light	Light Pin 223	18.9319560	72.8211440	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
52757080-834a-40fc-aa25-d422b659d172	street_light	Light Pin 224	18.9197790	72.8267630	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
bd636627-5c1d-4dc1-a215-ff5a9761d77f	street_light	Light Pin 225	18.9350260	72.8349740	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
bb8f0bef-4682-4fae-90ec-bcc29de47104	street_light	Light Pin 226	18.9470910	72.8209640	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2d468d1d-db95-48f5-a520-f2b64262395c	street_light	Light Pin 227	19.0524560	72.8589380	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
cb344905-afc2-4200-a76d-3162d6d71fb6	street_light	Light Pin 228	19.1981280	72.8459720	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5d0cddb7-9401-4288-a6b0-b5b25f0a8774	street_light	Light Pin 229	19.2182570	72.8724800	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
94abcf4b-5325-45bb-ad27-500dc199e7aa	street_light	Light Pin 230	19.0083380	72.8308350	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
8ac91d52-b1b5-4514-8fcf-42b2e16da2f2	street_light	Light Pin 231	18.9302840	72.8275360	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
52d95ca0-94b9-42e9-a875-2a52fcf1cd2e	street_light	Light Pin 232	18.9646180	72.8236110	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
5a41a576-006f-401f-abff-f2d3b0df4dec	street_light	Light Pin 233	19.0889980	72.8539040	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
20edf122-e865-4f9b-b254-f4b2b57a5e5c	street_light	Light Pin 234	19.0941220	72.8333510	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2a3687f1-8e4b-4ade-9b35-b9bfa27f2601	street_light	Light Pin 235	18.9153940	72.8237930	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
440a1a87-8ec6-4bca-9d5e-529acee6a8f2	street_light	Light Pin 236	19.1810440	72.8554360	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ef2ba053-6ddf-4bc3-91d7-f9c746aa6233	street_light	Light Pin 237	18.9444880	72.8238130	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ff7c0c89-b7b6-4b6f-8614-a4896fc9f8e7	street_light	Light Pin 238	19.0749700	72.8549600	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
1fe90367-6b7d-400b-b1f7-6b5630dc55ce	street_light	Light Pin 239	19.0855040	72.9130580	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
c363da4b-4b71-4466-8e33-27efa93b3371	street_light	Light Pin 240	18.9940040	72.8481450	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
bdea025a-d6d5-492e-9498-5b44fe13e218	street_light	Light Pin 241	18.9654920	72.8249000	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
effa9c43-3e3b-427e-beed-406d81b7b5cf	street_light	Light Pin 242	19.1457310	72.8299320	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
a2fff726-5739-40f4-99f2-8b08134b2650	street_light	Light Pin 243	18.9943570	72.8388260	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2538e5d9-107e-45af-93c1-a07b5520e988	street_light	Light Pin 244	19.2114580	72.9209980	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
2d3a53b8-7a6c-497c-89bb-ad84e02595dc	street_light	Light Pin 245	18.9220040	72.8184670	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
150cef3a-0c2b-4ce3-8c61-cf32a3acc4a4	street_light	Light Pin 246	18.9374430	72.8287880	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
ef629bba-1896-4f2e-9174-d16a0f9b2420	street_light	Light Pin 247	18.9249950	72.8200550	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
8d9216ea-5f73-405c-82ff-3d627ae742a5	street_light	Light Pin 248	19.1272300	72.8964470	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
4eb69cac-5dfa-4835-8895-03f15efded88	street_light	Light Pin 249	19.1883190	72.9171330	{"status": "ON"}	utilities	2026-07-21 22:15:53.602237
379e507c-f11a-4991-aa17-d0f39fac7ebd	transformer	Grid Node 0	19.0894730	72.8533420	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
9dacc794-aa18-4686-bf0f-505a746951c5	transformer	Grid Node 1	19.1491120	72.8541550	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
d909d98c-7ef5-49ea-944d-efde2f4c80bb	transformer	Grid Node 2	18.9272360	72.8261190	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
c0da0ecb-502f-48f2-b8c7-991cf4a5068c	transformer	Grid Node 3	18.9805490	72.8252120	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
8b9a4546-43de-4822-b91b-860feff60ebe	transformer	Grid Node 4	19.0273040	72.8484120	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
d17611d6-fe7e-41a7-82d0-19cc5f8116c7	transformer	Grid Node 5	19.2061870	72.8413090	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
4cca509e-0d29-4ff2-be8e-f7a32436a5d7	transformer	Grid Node 6	19.1001220	72.8537600	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
4b96597d-655c-4a4f-8667-f4d4fb08ec4a	transformer	Grid Node 7	19.2080940	72.8903010	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
067d8aec-e4ec-4c80-97b4-c0e836552453	transformer	Grid Node 8	19.0988350	72.8764440	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
ee0675f9-e838-4611-94e2-4e26d53ee79b	transformer	Grid Node 9	19.0245580	72.8502820	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
3ef878c7-612c-4181-83d1-296fabf0f4b3	transformer	Grid Node 10	19.0022410	72.8337190	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
102a352d-dfdf-46ee-b3cd-0aa50bf46208	transformer	Grid Node 11	18.9401240	72.8317010	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
44473483-217b-4091-a4f0-f2b066a4594c	transformer	Grid Node 12	18.9910140	72.8327610	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
f9fb84b7-54e0-4500-9f5d-be8b2faf070e	transformer	Grid Node 13	18.9375380	72.8278230	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
3c2b648d-a609-4ad5-8b34-ca644b260b2b	transformer	Grid Node 14	19.1821570	72.8539970	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
2969df41-5802-48c0-b23a-4ce5bad913a6	transformer	Grid Node 15	19.0042570	72.8488500	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
13e669df-af76-46ca-95d5-791380bb5cfd	transformer	Grid Node 16	19.0774190	72.8984680	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
db2e72c1-4d4f-47cd-97d9-65d0e8a572b5	transformer	Grid Node 17	19.0659170	72.9130910	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
cb0fd162-0410-43bb-be99-950b7abf8753	transformer	Grid Node 18	18.9699500	72.8328570	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
84b23f4d-2e17-474c-8c71-cd8d5b0c8b01	transformer	Grid Node 19	19.0263560	72.8498060	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
3ec9bf86-2ba3-411f-8ef1-c8b19aa55652	transformer	Grid Node 20	19.1113920	72.8261180	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
5b1b7084-26a3-4ff5-8777-4032689da2a4	transformer	Grid Node 21	19.1450910	72.8533970	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
da3793a3-47f8-4db5-9e0d-71dc19034798	transformer	Grid Node 22	18.9652980	72.8240230	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
068fdff3-146b-4517-8def-5d10d22810f3	transformer	Grid Node 23	18.9908940	72.8471080	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
138cee1b-1a74-4d4c-af13-39b4bd499767	transformer	Grid Node 24	19.0827480	72.8383430	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
c90fd283-5b78-401e-8bd6-ab0cd8213b7f	transformer	Grid Node 25	19.0598600	72.8534080	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
ae3c952f-33e2-428c-8cb6-3f322bb71dee	transformer	Grid Node 26	19.1635870	72.9262550	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
94582f83-5141-4d7e-9b7c-ea54e80b2bba	transformer	Grid Node 27	19.0554020	72.8379580	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
28a35160-5f01-47a4-8c22-4fb1f8321895	transformer	Grid Node 28	18.9296100	72.8156570	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
6e01ad54-4b8e-430d-8638-1565c4ba6307	transformer	Grid Node 29	19.1158870	72.9044700	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
ca5b13d8-2705-4f25-9733-ebb4a145af38	transformer	Grid Node 30	18.9536120	72.8177380	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
74e3627f-615e-4482-be4f-6ecee7bde292	transformer	Grid Node 31	19.1561230	72.8459200	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
5e875f6c-cc44-4c4e-9d64-4d05f45b9f5e	transformer	Grid Node 32	18.9518090	72.8343410	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
50b4c9ec-bf6e-4b7c-ad0f-2c5a13067f9f	transformer	Grid Node 33	18.9929790	72.8282460	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
7dbce392-8884-4a0a-9747-cf79213b46e6	transformer	Grid Node 34	19.1490060	72.8269610	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
03bed39f-a18b-4e98-80ae-08c4c3b7be1f	transformer	Grid Node 35	18.9526310	72.8345470	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
d4d2d8bc-6c5f-4759-b119-6476dc872e0b	transformer	Grid Node 36	18.9494210	72.8269520	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
4c02a5e2-1a0a-49eb-831d-4be0c9df9a26	transformer	Grid Node 37	19.0555680	72.8565970	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
ed4554d9-a47b-4b79-bd45-aec0d8db51a9	transformer	Grid Node 38	19.0013700	72.8498420	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
a3cf12e3-c396-4a3d-994c-3b11fdf6c22e	transformer	Grid Node 39	19.1262340	72.9234580	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
68501bec-abfb-4cba-ac32-a9d4a67bf3d0	transformer	Grid Node 40	19.1168580	72.8357970	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
1e69354f-26b5-4b88-a067-6149d9e0c3aa	transformer	Grid Node 41	19.1049190	72.8804900	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
32ddeb11-9a22-4c22-88ef-29158a3c17a3	transformer	Grid Node 42	18.9562540	72.8318920	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
df04c8d0-6ca5-408a-be46-5bafd542b344	transformer	Grid Node 43	18.9331460	72.8272380	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
e522a225-fb09-4f65-aa9b-f8a90d9247c4	transformer	Grid Node 44	19.0108900	72.8334000	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
c6998e65-fc27-435d-bc6a-7b736670d2c9	transformer	Grid Node 45	18.9554350	72.8189120	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
34feb013-2aef-44e7-9e46-edbda1848c09	transformer	Grid Node 46	18.9880060	72.8457060	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
c1a2792d-e24a-4202-a744-7fc15327aa14	transformer	Grid Node 47	18.9807940	72.8468800	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
1aac63dd-7fa0-49f3-bc9b-776babf78981	transformer	Grid Node 48	19.1190520	72.8292240	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
ba184074-a3f7-446d-b900-8c8b0fa355c4	transformer	Grid Node 49	18.9503050	72.8339090	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
7d5141aa-6435-438c-81f5-3dcb9c4e64db	transformer	Grid Node 50	18.9480470	72.8301590	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
5440ccf9-7713-4164-9418-d6b89e9414cd	transformer	Grid Node 51	19.1285970	72.8521620	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
e0f6daf6-2942-4307-b279-0205025c4eeb	transformer	Grid Node 52	19.2012660	72.8913810	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
44468325-9a4f-4602-bd13-3dfb77f91a0a	transformer	Grid Node 53	19.1913760	72.9024860	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
b75fb8eb-49ad-4008-8c13-e53583e4579d	transformer	Grid Node 54	18.9357110	72.8309790	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
3c3e6017-0d92-4009-a464-be5a16d31606	transformer	Grid Node 55	19.0288660	72.8533920	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
636c30eb-4068-48ce-ba99-c7d4e095f657	transformer	Grid Node 56	19.0665480	72.8503670	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
f0b21a6a-6b6f-4b45-a5a5-f9e54f49b37a	transformer	Grid Node 57	18.9843610	72.8483300	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
68dd5d29-24c9-43f4-8718-bd0b6068235f	transformer	Grid Node 58	18.9833780	72.8351500	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
cd83e053-49bd-4b7e-ba62-d6b4677b8ebe	transformer	Grid Node 59	19.1664100	72.8797930	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
1a32d780-71c3-4ef5-a7c6-f73761207476	transformer	Grid Node 60	18.9192620	72.8307580	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
e304adbc-a986-4040-a8b5-044238543ed3	transformer	Grid Node 61	19.0138040	72.8340720	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
ae36d50e-a004-4bb7-a760-031ea3803ce6	transformer	Grid Node 62	18.9608660	72.8163690	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
797cc9c9-61cb-4e79-aca3-c6545650ac69	transformer	Grid Node 63	18.9995340	72.8539810	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
54e04240-0488-4ead-9a01-290d998ede48	transformer	Grid Node 64	18.9402350	72.8313350	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
002e7d18-9f33-45a1-a387-7bcbda309488	transformer	Grid Node 65	19.1772730	72.9076940	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
28e7ce0b-ae20-4c24-afbf-0f33c41c3776	transformer	Grid Node 66	18.9212460	72.8365120	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
4413d4c2-8382-42a8-93fe-b4cd9b30bdef	transformer	Grid Node 67	19.0827900	72.9117470	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
eae0c8d7-5ffb-4ca1-b795-bfde01b9254e	transformer	Grid Node 68	19.2022610	72.8814000	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
442238da-112b-47de-b8a8-6ea5a0381850	transformer	Grid Node 69	19.1905290	72.8513200	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
42e526af-266e-4e78-bc77-0e1455b67bc4	transformer	Grid Node 70	18.9806160	72.8358450	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
2a99d8ee-9420-44de-8e91-9d114641f61c	transformer	Grid Node 71	19.1118020	72.9069960	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
94bc4afe-dd9a-4358-bf66-bc7457acd61d	transformer	Grid Node 72	18.9692770	72.8274240	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
62d7150b-2bef-48e3-94ea-41ec4b3f1fbd	transformer	Grid Node 73	19.0051320	72.8546390	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
f44ed000-7d36-4d19-9d65-b594bad68e41	transformer	Grid Node 74	19.1299200	72.8319770	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
1bd8efd8-feab-447f-aa87-20ae3ecb2f02	transformer	Grid Node 75	18.9898370	72.8332920	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
b4a8fd43-7a5b-4d85-8368-ea6d8348a060	transformer	Grid Node 76	19.0988550	72.8545620	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
bad27bda-79a9-45b6-8b99-10a55752b2bc	transformer	Grid Node 77	18.9227920	72.8233590	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
d85e20b6-1639-4d0f-9490-ad96e6e967ee	transformer	Grid Node 78	18.9911550	72.8400970	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
837f18bf-cca5-42ff-929d-a76c04e49d35	transformer	Grid Node 79	18.9570180	72.8335470	{"load": "74%"}	utilities	2026-07-21 22:15:53.602237
432b6954-fd58-4f59-9810-c0e2554f2d46	water_sensor	Water Valve 0	18.9818600	72.8294770	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
14b75c24-237e-4ad5-bcee-56f1b74775f4	water_sensor	Water Valve 1	19.1240390	72.9179700	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
36be9ee1-ab62-405b-bec4-ce138adc2f3e	water_sensor	Water Valve 2	19.1928610	72.8488710	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
34ffea6a-294e-483f-a300-ba6106023bcb	water_sensor	Water Valve 3	19.1164930	72.8458240	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
bf7aa622-a10b-4583-b851-bbafa7db5a87	water_sensor	Water Valve 4	18.9307130	72.8153370	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
a6443958-4b98-4409-a041-aec55c12dbbd	water_sensor	Water Valve 5	18.9577570	72.8237150	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
a3711b38-5dd4-4f3e-b38c-c0edecdb1809	water_sensor	Water Valve 6	19.0115130	72.8270840	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
70658887-b57d-4881-827f-bdaded4b7a3d	water_sensor	Water Valve 7	19.1634310	72.8336530	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
51d42912-2c96-4787-83f5-48ab8881b752	water_sensor	Water Valve 8	19.0488810	72.8566450	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
fad7d42d-2805-485f-a07c-b741e7c1be46	water_sensor	Water Valve 9	18.9469320	72.8350770	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
04b12d65-bba9-4d44-bc2a-e6e19554415d	water_sensor	Water Valve 10	18.9559430	72.8247190	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
7e7e0767-75f0-4461-a350-bed098b76318	water_sensor	Water Valve 11	19.2070310	72.8893970	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
bb57f7da-78ca-4d86-917a-f3ec2b632090	water_sensor	Water Valve 12	19.1736390	72.8793050	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
46c27b76-a0e6-4b6e-a915-6a17acbdc4a0	water_sensor	Water Valve 13	19.0831810	72.9103660	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
bc74dad6-b08a-4edc-8887-93aba827198a	water_sensor	Water Valve 14	19.0095120	72.8476480	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
be6b3bad-7184-49a5-90bb-0aeb09a1e190	water_sensor	Water Valve 15	19.0875600	72.8343550	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
dd108415-aa8b-4e35-a857-bcd2c533624e	water_sensor	Water Valve 16	19.0891880	72.8490370	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
429d60a3-4035-454a-bcc9-62ca5ba6d95b	water_sensor	Water Valve 17	18.9852620	72.8478420	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
2de481d5-894a-4a22-8492-39c5a3f240d9	water_sensor	Water Valve 18	19.1564710	72.8313100	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
bb6102fc-f79e-41c8-976c-08c0dfb0ce1f	water_sensor	Water Valve 19	18.9584560	72.8191970	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
5f130599-bf1f-47ab-87dd-efcd462fa9d1	water_sensor	Water Valve 20	19.1969610	72.8881940	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
46679387-8dc2-4d04-aa0d-63ee7e56fd31	water_sensor	Water Valve 21	19.1540640	72.9296780	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
29208100-b154-475a-bbfe-7baa8808bc44	water_sensor	Water Valve 22	19.1074200	72.9031290	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
c5682eeb-ef75-48c3-b790-37f154af806f	water_sensor	Water Valve 23	18.9677040	72.8312450	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
78471d26-196f-4dd3-9092-17691ae26f50	water_sensor	Water Valve 24	18.9691630	72.8349160	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
99ddbe97-1dc0-429f-8b4c-be4684e71ed0	water_sensor	Water Valve 25	18.9611260	72.8166240	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
762dd81e-9de1-451a-ab2f-ed78728f79d3	water_sensor	Water Valve 26	18.9980800	72.8423410	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
187ffdc4-2df6-4206-af17-08c31d924094	water_sensor	Water Valve 27	18.9574000	72.8378830	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
5e0dac91-9171-4ef0-9b38-b9078e9de7c5	water_sensor	Water Valve 28	18.9840700	72.8467670	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
bbe9fd30-8998-475f-b422-4bf01c3e0116	water_sensor	Water Valve 29	18.9268760	72.8225340	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
019ad1f7-f030-4947-8c8d-c060672b0a85	water_sensor	Water Valve 30	18.9504730	72.8365110	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
8df676f7-0054-4a2c-b598-d4e72834172c	water_sensor	Water Valve 31	18.9648370	72.8347990	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
ddec0f0c-a711-4b2c-b048-0b237bc78751	water_sensor	Water Valve 32	18.9975050	72.8407040	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
304f58cd-b70b-42d3-81e0-fb40857a1814	water_sensor	Water Valve 33	19.1705600	72.8535160	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
88699051-07ed-4f22-92a8-ee7eb1d71559	water_sensor	Water Valve 34	18.9959700	72.8413090	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
8854dff5-2a3c-4aca-baba-6dbe9e527e9f	water_sensor	Water Valve 35	19.1658940	72.8441990	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
b9c23c2d-7ad7-471b-9df4-a8e708f52f3f	water_sensor	Water Valve 36	18.9496190	72.8379910	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
0531c009-abcb-44b2-93d7-bb25ad996ad2	water_sensor	Water Valve 37	19.1497170	72.8844100	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
a113c6bd-2600-4bf3-a5f6-f40fe2f28f52	water_sensor	Water Valve 38	19.0886110	72.9074680	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
fd478a83-f11a-4bc7-b11f-3e3f7cb4fb87	water_sensor	Water Valve 39	18.9259000	72.8334440	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
69af3a91-548e-491c-a7c4-39464815704c	water_sensor	Water Valve 40	19.1590170	72.9136670	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
a4deb8fe-30b6-4a5f-98e0-d0ab03e1be8c	water_sensor	Water Valve 41	19.0147050	72.8289150	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
02e10dca-d229-4827-870e-5f22ffaa5f35	water_sensor	Water Valve 42	19.0449180	72.8484910	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
13eda0bb-6a71-44af-9da2-5a345a13f51f	water_sensor	Water Valve 43	19.1193240	72.9105560	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
e827d507-16f7-4929-8ab5-a9b8138924b2	water_sensor	Water Valve 44	18.9492220	72.8340330	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
3c0e1358-90b5-417e-aa7c-3be1393d6cef	water_sensor	Water Valve 45	19.1779520	72.8266960	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
0d6521a8-c1eb-44e4-88f8-d7b9e0745e07	water_sensor	Water Valve 46	19.0770650	72.9150470	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
d42a1dde-cce1-4315-9aaf-f31bf4850a49	water_sensor	Water Valve 47	19.1240230	72.8740020	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
f48823c3-a68b-4ea3-b419-0242004df61e	water_sensor	Water Valve 48	19.0618480	72.8496690	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
8749e96a-023a-4fbf-b3dc-074b7fee49b9	water_sensor	Water Valve 49	18.9404650	72.8194130	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
85c1e6b4-4985-49ea-90bb-da6edd1333c1	water_sensor	Water Valve 50	18.9905590	72.8459490	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
511fff74-c811-44ec-bd3a-47f49504e24c	water_sensor	Water Valve 51	18.9690750	72.8255010	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
9edc372e-66b5-4deb-889e-ad4a49f7a058	water_sensor	Water Valve 52	18.9485970	72.8253160	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
89999ba1-d4ac-4dbd-92d5-157c5c10b9b3	water_sensor	Water Valve 53	19.1797200	72.9154830	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
81303550-9b55-4e2b-a162-90925b4f0675	water_sensor	Water Valve 54	18.9835040	72.8404560	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
8fa32aa2-4f56-4a1e-affb-8d8f0798f8a2	water_sensor	Water Valve 55	18.9951940	72.8426420	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
d99640a5-0c96-4e4a-9827-d058d9da4c7b	water_sensor	Water Valve 56	19.1444850	72.8312150	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
35f6c447-12d3-4092-a2e4-45f42c24bb92	water_sensor	Water Valve 57	19.1068450	72.8486800	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
fcbe6e35-1efb-42e2-9ff9-29d15a9e78b8	water_sensor	Water Valve 58	19.0264530	72.8315740	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
487426a2-ba2d-44af-8055-82d0e2379ab9	water_sensor	Water Valve 59	19.1170750	72.8361620	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
e084b6d0-bc4b-4a54-b6ff-7f1080c8fa1a	water_sensor	Water Valve 60	19.2001030	72.8908970	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
186c9bc5-47f5-485c-8cd8-bc6fb30faff7	water_sensor	Water Valve 61	19.0031610	72.8369210	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
8157c82d-30e2-485d-be54-f684ffb636bf	water_sensor	Water Valve 62	19.1278200	72.9207710	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
485a4698-f9bf-480a-8c6a-675700b6dfae	water_sensor	Water Valve 63	19.0903220	72.8436210	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
67779844-a1ea-4349-a4f5-1898c3f6953d	water_sensor	Water Valve 64	19.2245890	72.8508460	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
d31f2c0b-4afa-446f-b980-a28ee6c279e3	water_sensor	Water Valve 65	19.0252940	72.8274190	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
97fa304b-12a0-41b4-80f2-67e69b63fc7d	water_sensor	Water Valve 66	19.2047680	72.8363970	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
629235de-b7a5-455e-9ca0-58a2dada1ccc	water_sensor	Water Valve 67	18.9932090	72.8422540	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
76d3401c-280d-4213-80d4-d95c74b04b92	water_sensor	Water Valve 68	19.1316800	72.8548190	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
a3bf074f-0839-4f95-b441-309b41207bb8	water_sensor	Water Valve 69	19.1371230	72.9172960	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
c14a0dd9-8252-48f8-b32a-ed10ada51ee6	water_sensor	Water Valve 70	19.0220810	72.8358450	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
8e3810ca-1e81-47dd-b290-f9a5f5d8e59e	water_sensor	Water Valve 71	19.0131220	72.8521600	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
27067092-b655-4e93-a423-03f59a468e34	water_sensor	Water Valve 72	19.0057670	72.8313300	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
79dca799-39e7-459e-8f1a-595d5bd10289	water_sensor	Water Valve 73	18.9920540	72.8492600	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
c83ad257-6a57-4e66-9337-8bc2c80174d7	water_sensor	Water Valve 74	19.1383700	72.8846610	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
d7f41c8b-1382-404b-ae95-c85962dfe732	water_sensor	Water Valve 75	19.2154350	72.9221150	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
478938c5-9c9f-4e1a-aab7-78f5cac1ab7d	water_sensor	Water Valve 76	19.0086720	72.8307580	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
362e65e7-8065-46be-93f6-434737571120	water_sensor	Water Valve 77	18.9831250	72.8303970	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
053f081d-c656-4ba8-90b6-81a2412a7a9c	water_sensor	Water Valve 78	19.2127010	72.8576520	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
c43b8bed-b10e-495f-80ca-d5bc228a2863	water_sensor	Water Valve 79	18.9177320	72.8268160	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
5d260937-0f4a-4980-b25b-ecea13eb0add	water_sensor	Water Valve 80	19.2267050	72.8434060	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
5c02b7dd-a654-4114-a0ad-fb58c85b03a0	water_sensor	Water Valve 81	19.2153680	72.8412190	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
03ac5f25-c019-4b1f-a29f-0107966bcc4c	water_sensor	Water Valve 82	18.9222370	72.8345260	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
0984d803-a577-4f71-b4c1-b5ec5edca4cc	water_sensor	Water Valve 83	19.1900940	72.8261160	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
b08331bb-25ca-4659-a21a-d232340a2680	water_sensor	Water Valve 84	19.0195540	72.8465500	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
c8663f30-a6fb-4584-beed-66694fa90ad9	water_sensor	Water Valve 85	19.0732980	72.8512840	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
e292beb4-3063-4e61-8c7a-ac4b0e2a1d4a	water_sensor	Water Valve 86	19.0981570	72.8999560	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
fe99fbab-8cd0-4888-92b6-e7f483e8aed1	water_sensor	Water Valve 87	18.9868590	72.8447950	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
0a454d85-7d57-45f5-a33c-e2f0c7870b2b	water_sensor	Water Valve 88	19.0879180	72.9239140	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
764b6def-65ca-4809-aab5-aff228b304f1	water_sensor	Water Valve 89	19.0023260	72.8361720	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
758df631-2014-418f-a52d-946013053908	water_sensor	Water Valve 90	19.1452500	72.8996990	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
e49bcf29-cc89-41f9-8cd7-a04d60cec745	water_sensor	Water Valve 91	19.1210390	72.8905070	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
273790c8-cb63-4504-8aa0-be3458b177ae	water_sensor	Water Valve 92	19.0178980	72.8334890	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
e6c24daf-975c-4db2-b3e1-5f97b0ee0cc0	water_sensor	Water Valve 93	18.9664620	72.8348540	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
95a01123-9d0f-4b5c-8854-3bbcb1b16402	water_sensor	Water Valve 94	19.0063230	72.8389690	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
ea361186-c20a-4f0c-b15b-37769b399e93	water_sensor	Water Valve 95	19.0468550	72.8559200	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
015c9589-47d3-44c7-bc9f-45917e37dda5	water_sensor	Water Valve 96	19.1451110	72.9098130	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
46fefb68-933c-4ebe-939f-d96831172f13	water_sensor	Water Valve 97	19.1671400	72.8933410	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
ff1d46c9-3bfb-41cc-b37f-bea2ed96f61b	water_sensor	Water Valve 98	19.0795700	72.8455330	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
bfed4295-7507-4cec-8f4d-0f88ccd2ccdd	water_sensor	Water Valve 99	19.0285460	72.8341820	{"pressure": "Normal"}	utilities	2026-07-21 22:15:53.602237
0ab01dd8-d9fa-4d17-bd82-692cdd5883ec	weather_station	Mausam Node 0	19.1770360	72.8544250	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
a1eda672-2e6e-4026-b334-cfb23d8546c4	weather_station	Mausam Node 1	19.1076460	72.8811530	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
75137dbb-aa6b-4bc5-9013-471c813a1583	weather_station	Mausam Node 2	19.0619070	72.8785710	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
7300dae6-91b6-4f7a-905f-f94076819f78	weather_station	Mausam Node 3	18.9187010	72.8327720	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
790a0896-1b0a-4e38-9f8c-e138362352dd	weather_station	Mausam Node 4	18.9431780	72.8194680	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
8fef5195-2d74-444a-9829-a3b06340ad80	weather_station	Mausam Node 5	19.0003050	72.8397440	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
29ad9786-b921-47f3-84e7-45410f3466fa	weather_station	Mausam Node 6	19.1685980	72.8796750	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
edc9c386-9963-432f-aecc-7c786b3220f0	weather_station	Mausam Node 7	19.1007760	72.8401990	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
9ce25cba-efd1-446e-906a-50f77ed4730c	weather_station	Mausam Node 8	19.1085150	72.8837890	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
c0e29582-506e-45e0-bf4a-c95680dff83f	weather_station	Mausam Node 9	19.1363230	72.8582570	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
51ffbefa-c8f1-423a-ae8e-32b6606ad539	weather_station	Mausam Node 10	19.2066050	72.8568270	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
7ffa8bcf-eb21-489f-8034-7a324c9d96ff	weather_station	Mausam Node 11	19.0946660	72.9108560	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
bad63d71-94f2-4452-b022-2fa1f44dd0e4	weather_station	Mausam Node 12	18.9284030	72.8325150	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
f413ca1f-222c-44e1-a598-21f44da32b13	weather_station	Mausam Node 13	19.0736220	72.8801130	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
3e26cb11-239a-4e94-8958-1d62fef04032	weather_station	Mausam Node 14	18.9257130	72.8224540	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
2449ea80-bc97-455b-a972-386ba4887a7b	weather_station	Mausam Node 15	19.1045830	72.8981770	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
5b1f167b-c6ea-494a-a497-2fc48006b7a7	weather_station	Mausam Node 16	18.9249380	72.8360800	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
c436ac91-9989-442c-940a-b820d3cf59fd	weather_station	Mausam Node 17	18.9376070	72.8263900	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
77ba43ae-62b8-4493-9466-8ecdb9cc0bcf	weather_station	Mausam Node 18	19.2052190	72.9234290	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
8729b8ec-1552-4081-bf66-fbfcccac594f	weather_station	Mausam Node 19	19.0687560	72.8299320	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
a1f338f4-e26d-4d28-a16e-4deac3da7105	weather_station	Mausam Node 20	19.1875550	72.9271110	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
2b8ee5ba-d046-494c-a85a-c74e39bc30b1	weather_station	Mausam Node 21	19.1911020	72.9059500	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
81c7f788-bc17-4c3a-a17d-4c497580fcd0	weather_station	Mausam Node 22	18.9630010	72.8368890	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
3d8a1243-d721-4043-bcb5-aebb6c869bff	weather_station	Mausam Node 23	18.9511240	72.8266000	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
8ed1d1b5-6bf9-4118-a107-481ca5c619e7	weather_station	Mausam Node 24	19.0276910	72.8333940	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
8c64164a-ec12-4df3-b101-2dfcdc984da3	weather_station	Mausam Node 25	19.0007540	72.8348170	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
1249ac0b-1e1f-4d5d-9b99-143f0cd9f7b8	weather_station	Mausam Node 26	19.1286960	72.8460510	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
b851ee3c-61a8-4e13-8f0b-43342a14d6e7	weather_station	Mausam Node 27	18.9975810	72.8425110	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
b80ff206-2305-4885-b931-efb78ffe4bb6	weather_station	Mausam Node 28	18.9812560	72.8406980	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
8efcd0f6-3a72-4e6b-9926-a759355d0a2d	weather_station	Mausam Node 29	19.0630300	72.8383160	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
8386dd20-b1af-4951-862f-e475916d3530	weather_station	Mausam Node 30	19.0217370	72.8398400	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
3774fe5d-cddb-4496-9976-e5175f118a86	weather_station	Mausam Node 31	18.9500590	72.8315260	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
6eedec25-513b-4424-99a7-f28175fc1eba	weather_station	Mausam Node 32	19.0093190	72.8253370	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
c80eaaf7-0580-499d-8eba-30ec91e83d44	weather_station	Mausam Node 33	18.9314840	72.8291560	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
09222ad3-3728-4bc4-925b-5ada0434ab74	weather_station	Mausam Node 34	19.0160460	72.8404520	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
e598907c-50ee-462e-92ea-0576a86216c5	weather_station	Mausam Node 35	19.1661560	72.8344490	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
54d15d62-15c2-4e5a-8449-a3f76b74e8d0	weather_station	Mausam Node 36	19.0768210	72.8497400	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
f8a6098e-ff3c-4385-a2bc-9ef513e16a3d	weather_station	Mausam Node 37	18.9893090	72.8511110	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
f56349de-094a-4f2f-86e5-15b8d46cae5c	weather_station	Mausam Node 38	19.1785340	72.8765650	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
84e3544f-c74b-4b17-a316-b6cf991b2688	weather_station	Mausam Node 39	19.1846770	72.8345760	{"aqi": 110, "temp": "28C"}	weather	2026-07-21 22:15:53.602237
af67a23b-986a-4372-a405-eaecf1f88565	traffic_signal	Dadar Junction	19.0178000	72.8478000	{"reason": "High traffic demand", "is_demo": true, "red_time": 40, "assessment": "Potential congestion increase", "cycle_time": 70, "green_time": 30, "traffic_level": "SEVERE"}	traffic	2026-08-28 03:17:13.505291
978f1968-9211-4173-8c3f-02e5a7ef9742	traffic_signal	Wadala Junction	19.0163000	72.8587000	{"reason": "Heavy commercial outflow", "is_demo": true, "red_time": 35, "assessment": "Sustained high volumes from port access", "cycle_time": 70, "green_time": 35, "traffic_level": "HIGH"}	traffic	2026-08-28 03:17:13.505291
f051e67e-a2ce-4d43-a2e2-d48bb87e01b3	traffic_signal	Sion Circle	19.0390000	72.8619000	{"reason": "Merging bottleneck", "is_demo": true, "red_time": 45, "assessment": "Congestion cascading backwards", "cycle_time": 80, "green_time": 35, "traffic_level": "HIGH"}	traffic	2026-08-28 03:17:13.505291
786fd97f-5099-41b5-b5c9-d1fc89d1ed4e	traffic_signal	Matunga Junction	19.0270000	72.8550000	{"reason": "Standard local movement", "is_demo": true, "red_time": 30, "assessment": "Flow stabilizing", "cycle_time": 60, "green_time": 30, "traffic_level": "MODERATE"}	traffic	2026-08-28 03:17:13.505291
17a7ee5c-ab37-4fe4-b99e-013c4cea85d2	traffic_signal	Bandra Junction	19.0560000	72.8320000	{"reason": "Highway exit buildup", "is_demo": true, "red_time": 45, "assessment": "Clearance expected in 15 mins", "cycle_time": 90, "green_time": 45, "traffic_level": "HIGH"}	traffic	2026-08-28 03:17:13.505291
9ea42355-89ca-46f0-b58b-6a37714d64fd	traffic_signal	BKC Junction	19.0596000	72.8295000	{"reason": "Corporate sector outflow nominal", "is_demo": true, "red_time": 20, "assessment": "Continuously monitoring perimeter", "cycle_time": 60, "green_time": 40, "traffic_level": "LOW"}	traffic	2026-08-28 03:17:13.505291
894ff76c-a263-4ed0-9c00-5a3060440eba	traffic_signal	Worli Junction	19.0169000	72.8166000	{"reason": "Sea link approach traffic", "is_demo": true, "red_time": 35, "assessment": "Volumes within normal thresholds", "cycle_time": 70, "green_time": 35, "traffic_level": "MODERATE"}	traffic	2026-08-28 03:17:13.505291
6195470e-4031-4f89-8857-5ce348d50f04	traffic_signal	Hutatma Chowk Junction	18.9322000	72.8312000	{"area": "Fort", "road": "Mahatma Gandhi Road", "signal_code": "SIG-SOU-001"}	traffic_signal	2026-09-08 17:04:50.69266
5384a3cd-3839-42b6-bbf4-fb8d3c2fc0f7	traffic_signal	CSMT Junction	18.9402000	72.8356000	{"area": "Fort", "road": "Mahatma Gandhi Road", "signal_code": "SIG-SOU-002"}	traffic_signal	2026-09-08 17:04:50.69266
622338b9-b794-46c0-9b95-1ece33df3d8c	traffic_signal	Churchgate Junction	18.9320000	72.8270000	{"area": "Churchgate", "road": "Maharshi Karve Road", "signal_code": "SIG-SOU-003"}	traffic_signal	2026-09-08 17:04:50.69266
5883c94a-fa0f-4a78-9ee0-c5f9c4b273b7	traffic_signal	Marine Drive Junction	18.9437000	72.8236000	{"area": "Marine Drive", "road": "Netaji Subhash Chandra Bose Road", "signal_code": "SIG-SOU-004"}	traffic_signal	2026-09-08 17:04:50.69266
4407747f-e360-42c0-bfa6-df2afdc27c2a	traffic_signal	Nariman Point Junction	18.9256000	72.8242000	{"area": "Nariman Point", "road": "Nariman Point", "signal_code": "SIG-SOU-005"}	traffic_signal	2026-09-08 17:04:50.69266
ec145b30-c966-49a4-ab9b-3285f6695c02	traffic_signal	Haji Ali Junction	18.9827000	72.8118000	{"area": "Haji Ali", "road": "Dr Annie Besant Road", "signal_code": "SIG-SOU-006"}	traffic_signal	2026-09-08 17:04:50.69266
fde330ce-138f-4072-ae10-051435acfedc	traffic_signal	Mahalaxmi Junction	18.9822000	72.8234000	{"area": "Mahalaxmi", "road": "Dr E Moses Road", "signal_code": "SIG-SOU-007"}	traffic_signal	2026-09-08 17:04:50.69266
536ccf71-174d-451d-ad0c-d6143c692292	traffic_signal	Tardeo Junction	18.9720000	72.8160000	{"area": "Tardeo", "road": "Tardeo Road", "signal_code": "SIG-SOU-008"}	traffic_signal	2026-09-08 17:04:50.69266
0773e544-c4b3-405f-afdc-e76f70f8af09	traffic_signal	Mumbai Central Junction	18.9697000	72.8195000	{"area": "Mumbai Central", "road": "Bellasis Road", "signal_code": "SIG-SOU-009"}	traffic_signal	2026-09-08 17:04:50.69266
0eb75675-e91f-4484-90b4-f0bc2cbb6cd7	traffic_signal	Byculla Junction	18.9766000	72.8328000	{"area": "Byculla", "road": "Dr Babasaheb Ambedkar Road", "signal_code": "SIG-SOU-010"}	traffic_signal	2026-09-08 17:04:50.69266
bc276040-2430-41fb-b1ba-2cecf867a4e2	traffic_signal	Worli Naka Junction	19.0178000	72.8173000	{"area": "Worli", "road": "Annie Besant Road", "signal_code": "SIG-WOR-001"}	traffic_signal	2026-09-08 17:04:50.69266
19ddd422-42ce-4da2-bace-7a3335b3b4cc	traffic_signal	Worli Sea Face Junction	19.0084000	72.8170000	{"area": "Worli", "road": "Veer Savarkar Marg", "signal_code": "SIG-WOR-002"}	traffic_signal	2026-09-08 17:04:50.69266
96e5b191-b4f4-4dd1-9843-ae18b746fc19	traffic_signal	Prabhadevi Junction	19.0140000	72.8290000	{"area": "Prabhadevi", "road": "Gokhale Road", "signal_code": "SIG-WOR-003"}	traffic_signal	2026-09-08 17:04:50.69266
8e9bff0f-f9d6-4318-9030-555a7c097969	traffic_signal	Lower Parel Junction	18.9986000	72.8257000	{"area": "Lower Parel", "road": "Senapati Bapat Marg", "signal_code": "SIG-WOR-004"}	traffic_signal	2026-09-08 17:04:50.69266
c6cfb255-4dd2-4223-a1c3-79170c77ecf5	traffic_signal	Elphinstone Junction	19.0015000	72.8258000	{"area": "Elphinstone", "road": "Senapati Bapat Marg", "signal_code": "SIG-WOR-005"}	traffic_signal	2026-09-08 17:04:50.69266
33638101-04ca-4bed-839f-7702216108ba	traffic_signal	Dadar TT Circle	19.0178000	72.8478000	{"area": "Dadar", "road": "Dr Babasaheb Ambedkar Road", "signal_code": "SIG-DAD-001"}	traffic_signal	2026-09-08 17:04:50.69266
a7d603ae-72c4-4905-b0db-e012d89a2e46	traffic_signal	Dadar Shivaji Park Junction	19.0282000	72.8408000	{"area": "Dadar", "road": "Veer Savarkar Marg", "signal_code": "SIG-DAD-002"}	traffic_signal	2026-09-08 17:04:50.69266
c5980fde-e36e-4c20-b660-febb435cbc54	traffic_signal	Mahim Junction	19.0418000	72.8408000	{"area": "Mahim", "road": "L.J. Road", "signal_code": "SIG-MAH-001"}	traffic_signal	2026-09-08 17:04:50.69266
37c451e1-2bcf-4202-8d16-7b33fe9b8abc	traffic_signal	Mahim Causeway Junction	19.0437000	72.8397000	{"area": "Mahim", "road": "Mahim Causeway", "signal_code": "SIG-MAH-002"}	traffic_signal	2026-09-08 17:04:50.69266
d803906e-7f3e-4276-847a-2c1137d7b38c	traffic_signal	Matunga Five Gardens Junction	19.0273000	72.8554000	{"area": "Matunga", "road": "King Circle", "signal_code": "SIG-MAT-001"}	traffic_signal	2026-09-08 17:04:50.69266
39106148-7a6b-4be2-8941-a1c0f8b38338	traffic_signal	Sion Circle	19.0436000	72.8616000	{"area": "Sion", "road": "Dr Babasaheb Ambedkar Road", "signal_code": "SIG-SIO-001"}	traffic_signal	2026-09-08 17:04:50.69266
cddef124-03d6-4080-ba9e-5d0d13186e04	traffic_signal	Sion Hospital Junction	19.0447000	72.8610000	{"area": "Sion", "road": "Dr Babasaheb Ambedkar Road", "signal_code": "SIG-SIO-002"}	traffic_signal	2026-09-08 17:04:50.69266
cd8353e5-35f3-4908-8bd8-7469bea036f7	traffic_signal	Bandra Junction	19.0544000	72.8406000	{"area": "Bandra", "road": "S.V. Road", "signal_code": "SIG-BAN-001"}	traffic_signal	2026-09-08 17:04:50.69266
d917ad96-6f35-4d74-88f3-145bce83bcd1	traffic_signal	Bandra Bandstand Junction	19.0425000	72.8196000	{"area": "Bandra", "road": "Bandstand Road", "signal_code": "SIG-BAN-002"}	traffic_signal	2026-09-08 17:04:50.69266
0f254671-b2d1-4595-833d-f5c6ccbdca98	traffic_signal	Khar Junction	19.0680000	72.8363000	{"area": "Khar", "road": "S.V. Road", "signal_code": "SIG-BAN-003"}	traffic_signal	2026-09-08 17:04:50.69266
ad66ef0b-eb35-438a-a350-e39aef911b67	traffic_signal	Santacruz Junction	19.0812000	72.8416000	{"area": "Santacruz", "road": "S.V. Road", "signal_code": "SIG-SAN-001"}	traffic_signal	2026-09-08 17:04:50.69266
b9bd40cb-44cc-45f7-a00d-2197ce85ec0b	traffic_signal	Vakola Junction	19.0794000	72.8560000	{"area": "Santacruz", "road": "Western Express Highway", "signal_code": "SIG-SAN-002"}	traffic_signal	2026-09-08 17:04:50.69266
bb879fcb-3700-41a4-a25a-6fcfe4cce058	traffic_signal	Milan Subway Junction	19.0946000	72.8425000	{"area": "Santacruz", "road": "S.V. Road", "signal_code": "SIG-SAN-003"}	traffic_signal	2026-09-08 17:04:50.69266
b50d7246-f7f4-4e73-a13a-373c4faf2a3b	traffic_signal	BKC Kalanagar Junction	19.0655000	72.8660000	{"area": "BKC", "road": "Bandra Kurla Complex Road", "signal_code": "SIG-BKC-001"}	traffic_signal	2026-09-08 17:04:50.69266
b40fdc72-96e2-4048-9a69-5086b61fb44a	traffic_signal	BKC Bharat Diamond Bourse Junction	19.0607000	72.8680000	{"area": "BKC", "road": "BKC Road", "signal_code": "SIG-BKC-002"}	traffic_signal	2026-09-08 17:04:50.69266
e49660f6-23f0-4f98-998e-5692e62dba62	traffic_signal	BKC Kurla Junction	19.0675000	72.8717000	{"area": "BKC", "road": "BKC Connector", "signal_code": "SIG-BKC-003"}	traffic_signal	2026-09-08 17:04:50.69266
28ec1ca1-fc65-45fa-ad45-018c15001636	traffic_signal	Andheri Station Junction	19.1197000	72.8468000	{"area": "Andheri", "road": "S.V. Road", "signal_code": "SIG-AND-001"}	traffic_signal	2026-09-08 17:04:50.69266
b2d15ebb-886b-497f-950a-17cc46ae430a	traffic_signal	Andheri Market Junction	19.1192000	72.8479000	{"area": "Andheri", "road": "Andheri Ghatkopar Road", "signal_code": "SIG-AND-002"}	traffic_signal	2026-09-08 17:04:50.69266
5691f2d5-fbe4-4c22-a9c1-3d5ef0a7ebb8	traffic_signal	Andheri Kurla Junction	19.1156000	72.8720000	{"area": "Andheri", "road": "Andheri-Kurla Road", "signal_code": "SIG-AND-003"}	traffic_signal	2026-09-08 17:04:50.69266
99513606-d20e-4938-9646-417b96a52a80	traffic_signal	Chakala Junction	19.1075000	72.8787000	{"area": "Andheri", "road": "Andheri-Kurla Road", "signal_code": "SIG-AND-004"}	traffic_signal	2026-09-08 17:04:50.69266
117791e3-f16d-4c09-bdec-ad2514c31cde	traffic_signal	Marol Naka Junction	19.1087000	72.8793000	{"area": "Marol", "road": "Andheri-Kurla Road", "signal_code": "SIG-AND-005"}	traffic_signal	2026-09-08 17:04:50.69266
cb673f52-23d2-43e5-8333-383c518c8ee7	traffic_signal	Saki Naka Junction	19.1030000	72.8870000	{"area": "Saki Naka", "road": "Saki Vihar Road", "signal_code": "SIG-AND-006"}	traffic_signal	2026-09-08 17:04:50.69266
9bfcaf41-1f05-47a7-a55b-c4206b504378	traffic_signal	Jogeshwari Junction	19.1364000	72.8479000	{"area": "Jogeshwari", "road": "S.V. Road", "signal_code": "SIG-JOG-001"}	traffic_signal	2026-09-08 17:04:50.69266
85167a8d-e080-4bb1-ad94-d84e00bc8912	traffic_signal	JVLR Western Junction	19.1375000	72.8470000	{"area": "Jogeshwari", "road": "JVLR", "signal_code": "SIG-JOG-002"}	traffic_signal	2026-09-08 17:04:50.69266
80b13ccf-8b0a-49a4-9903-9a5f9739037c	traffic_signal	Goregaon Junction	19.1663000	72.8526000	{"area": "Goregaon", "road": "S.V. Road", "signal_code": "SIG-GOR-001"}	traffic_signal	2026-09-08 17:04:50.69266
19e54f28-8d6c-4a98-96c0-fc14b2e62590	traffic_signal	Goregaon WEH Junction	19.1551000	72.8492000	{"area": "Goregaon", "road": "Western Express Highway", "signal_code": "SIG-GOR-002"}	traffic_signal	2026-09-08 17:04:50.69266
8fc58d1f-e4ef-4e11-8326-5ccae1104aee	traffic_signal	Aarey Junction	19.1434000	72.8847000	{"area": "Aarey", "road": "JVLR", "signal_code": "SIG-GOR-003"}	traffic_signal	2026-09-08 17:04:50.69266
1c7d00d0-c315-4bdb-be93-545d408ef92f	traffic_signal	Malad Junction	19.1860000	72.8481000	{"area": "Malad", "road": "S.V. Road", "signal_code": "SIG-MAL-001"}	traffic_signal	2026-09-08 17:04:50.69266
b4fa4333-7ccd-4973-8390-fb9c073f7538	traffic_signal	Malad Link Road Junction	19.1844000	72.8350000	{"area": "Malad", "road": "New Link Road", "signal_code": "SIG-MAL-002"}	traffic_signal	2026-09-08 17:04:50.69266
7eaa9651-3978-4d0c-93db-af1aa900006b	traffic_signal	Kandivali Junction	19.2055000	72.8422000	{"area": "Kandivali", "road": "S.V. Road", "signal_code": "SIG-KAN-001"}	traffic_signal	2026-09-08 17:04:50.69266
a295064f-9b20-4116-a083-a0a6db2af40f	traffic_signal	Borivali Junction	19.2307000	72.8567000	{"area": "Borivali", "road": "S.V. Road", "signal_code": "SIG-BOR-001"}	traffic_signal	2026-09-08 17:04:50.69266
20176dbe-c336-452d-b582-c33bc641e8ea	traffic_signal	Borivali WEH Junction	19.2300000	72.8550000	{"area": "Borivali", "road": "Western Express Highway", "signal_code": "SIG-BOR-002"}	traffic_signal	2026-09-08 17:04:50.69266
550c3b5d-c034-4f6f-8c11-ad2012618545	traffic_signal	Kurla Depot Junction	19.0726000	72.8794000	{"area": "Kurla", "road": "LBS Marg", "signal_code": "SIG-KUR-001"}	traffic_signal	2026-09-08 17:04:50.69266
67ca2e85-012b-40da-a559-24dd4f9ebc92	traffic_signal	Kurla Station Junction	19.0728000	72.8826000	{"area": "Kurla", "road": "LBS Marg", "signal_code": "SIG-KUR-002"}	traffic_signal	2026-09-08 17:04:50.69266
7adb3499-e4da-425a-b1be-52de6d8ceeeb	traffic_signal	Ghatkopar Junction	19.0860000	72.9081000	{"area": "Ghatkopar", "road": "LBS Marg", "signal_code": "SIG-GHA-001"}	traffic_signal	2026-09-08 17:04:50.69266
33751247-ce67-46e8-89dd-4dfd9199b295	traffic_signal	Ghatkopar West Junction	19.0865000	72.9089000	{"area": "Ghatkopar", "road": "J.M. Road", "signal_code": "SIG-GHA-002"}	traffic_signal	2026-09-08 17:04:50.69266
d371eb4a-fc63-4dd7-a75a-fd2012a1f201	traffic_signal	Vikhroli Junction	19.1115000	72.9273000	{"area": "Vikhroli", "road": "Eastern Express Highway", "signal_code": "SIG-VIK-001"}	traffic_signal	2026-09-08 17:04:50.69266
6fefba03-216d-48d1-bba6-1c7f89de2615	traffic_signal	Vikhroli Parksite Junction	19.1130000	72.9315000	{"area": "Vikhroli", "road": "LBS Marg", "signal_code": "SIG-VIK-002"}	traffic_signal	2026-09-08 17:04:50.69266
8a0f30a0-9057-47d3-8c05-07a252bfd48e	traffic_signal	Bhandup Junction	19.1437000	72.9376000	{"area": "Bhandup", "road": "LBS Marg", "signal_code": "SIG-BHA-001"}	traffic_signal	2026-09-08 17:04:50.69266
7ed2754f-d89d-434c-94ff-fb109924b9c7	traffic_signal	Mulund Check Naka	19.1726000	72.9617000	{"area": "Mulund", "road": "LBS Marg", "signal_code": "SIG-MUL-001"}	traffic_signal	2026-09-08 17:04:50.69266
a75b7c65-7524-49bd-9a37-3db44901a55d	traffic_signal	Chembur Junction	19.0626000	72.8973000	{"area": "Chembur", "road": "Sion Panvel Highway", "signal_code": "SIG-CHE-001"}	traffic_signal	2026-09-08 17:04:50.69266
65aa004f-a2b2-4eac-84a9-c09fb619a789	traffic_signal	Amar Mahal Junction	19.0472000	72.8950000	{"area": "Chembur", "road": "Eastern Express Highway", "signal_code": "SIG-CHE-002"}	traffic_signal	2026-09-08 17:04:50.69266
ce390c23-f218-4ea4-a68b-2a9caa84a093	traffic_signal	RCF Junction	19.0465000	72.9027000	{"area": "Chembur", "road": "Sion Panvel Road", "signal_code": "SIG-CHE-003"}	traffic_signal	2026-09-08 17:04:50.69266
f29ef3ee-b345-4bf8-9e69-039bb660b725	traffic_signal	Vashi Naka Junction	19.0218000	72.8902000	{"area": "Chembur", "road": "Sion Trombay Road", "signal_code": "SIG-CHE-004"}	traffic_signal	2026-09-08 17:04:50.69266
f20d118f-085d-4c93-a693-d2fa999ad809	traffic_signal	Wadala Junction	19.0176000	72.8580000	{"area": "Wadala", "road": "Wadala Truck Terminal Road", "signal_code": "SIG-WAD-001"}	traffic_signal	2026-09-08 17:04:50.69266
ac0f6750-b632-459c-ad10-f9c89e684605	traffic_signal	Wadala IMAX Junction	19.0167000	72.8692000	{"area": "Wadala", "road": "Anik Panjarapole Link Road", "signal_code": "SIG-WAD-002"}	traffic_signal	2026-09-08 17:04:50.69266
d5345fab-0739-4f93-9193-58afae8af705	traffic_signal	Antop Hill Junction	19.0324000	72.8755000	{"area": "Antop Hill", "road": "Dr Babasaheb Ambedkar Road", "signal_code": "SIG-WAD-003"}	traffic_signal	2026-09-08 17:04:50.69266
d2e6c6f9-e4c1-4902-968a-9f27699b3748	traffic_signal	Mankhurd Junction	19.0496000	72.9324000	{"area": "Mankhurd", "road": "Sion Panvel Highway", "signal_code": "SIG-MAN-001"}	traffic_signal	2026-09-08 17:04:50.69266
8a580f56-7a2a-4c15-a552-dce5a0e4f7d1	traffic_signal	Govandi Junction	19.0550000	72.9150000	{"area": "Govandi", "road": "Sion Panvel Road", "signal_code": "SIG-GOV-001"}	traffic_signal	2026-09-08 17:04:50.69266
\.


--
-- TOC entry 5290 (class 0 OID 25435)
-- Dependencies: 223
-- Data for Name: scenarios; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.scenarios (id, name, description, duration_seconds, created_at) FROM stdin;
5	multi_vehicle_heavy_rain.mp4	Heavy Rain -> Multi-Vehicle Accident -> Coordinated Response	120	2026-07-28 18:28:58.898986
6	commercial_building_fire.mp4	Commercial Building Fire -> Multi-Department Emergency Response	140	2026-07-28 19:07:26.547389
7	bank_robbery_pursuit.mp4	Bank Robbery -> Suspect Vehicle Escape -> AI Tracking -> Pursuit	120	2026-07-28 19:30:30.594585
\.


--
-- TOC entry 5292 (class 0 OID 25448)
-- Dependencies: 225
-- Data for Name: timeline_events; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.timeline_events (id, scenario_id, timestamp_second, event_type, payload, created_at) FROM stdin;
13	5	0	SIMULATION_START	{"status": "Nominal", "weather": "Clear"}	2026-07-28 18:28:58.907197
14	5	5	WEATHER_ALERT	{"weather": "Heavy Rain", "rainfall": "120mm", "visibility": "Low"}	2026-07-28 18:28:58.912865
15	5	10	TRAFFIC_UPDATE	{"density": "High", "congestion": "Increasing"}	2026-07-28 18:28:58.91365
16	5	22	ACCIDENT_DETECTED	{"cctvId": "CAM-04", "location": "Ring Road Junction", "severity": "Critical", "vehicles": ["Bus", "Car"]}	2026-07-28 18:28:58.914292
17	5	90	UTILITY_FAULT	{"asset": "Traffic Signal", "status": "Damaged"}	2026-07-28 18:28:58.914852
18	5	120	INCIDENT_RESOLVED	{"status": "Cleared", "roadOpen": true}	2026-07-28 18:28:58.915419
19	6	0	SIMULATION_START	{"grid": "Stable", "status": "Nominal"}	2026-07-28 19:07:26.550921
20	6	20	ANOMALY_DETECTED	{"sensor": "CCTV & Smoke Detector", "status": "Smoke Detected", "location": "Central Commercial Complex"}	2026-07-28 19:07:26.552994
21	6	30	FIRE	{"cctvId": "CAM-12", "location": "Central Commercial Complex", "severity": "Critical", "windSpeed": "15 km/h", "buildingType": "Commercial"}	2026-07-28 19:07:26.553564
22	6	130	FIRE_CONTAINED	{"status": "Intensity Decreasing", "smokeLevel": "Low"}	2026-07-28 19:07:26.554035
23	6	140	INCIDENT_RESOLVED	{"status": "Extinguished", "safetyCleared": true}	2026-07-28 19:07:26.554444
24	7	0	SIMULATION_START	{"status": "Nominal", "cityState": "Normal Operations"}	2026-07-28 19:30:30.598092
25	7	20	ANOMALY_DETECTED	{"sensor": "Silent Alarm", "status": "Robbery in Progress", "location": "Central City Bank"}	2026-07-28 19:30:30.600274
26	7	42	CRIME_REPORTED	{"cctvId": "CAM-12", "weapons": true, "location": "Central City Bank", "severity": "Critical", "suspects": 2, "trackingId": "TRK-8821", "vehiclesInvolved": ["Black SUV"]}	2026-07-28 19:30:30.600877
27	7	55	CCTV_TRACKING_UPDATE	{"cctvId": "CAM-18", "direction": "Northbound on Main St", "trackingId": "TRK-8821", "confidenceScore": "98%"}	2026-07-28 19:30:30.602501
28	7	65	CCTV_TRACKING_UPDATE	{"cctvId": "CAM-23", "direction": "Approaching Highway On-Ramp", "trackingId": "TRK-8821", "confidenceScore": "96%"}	2026-07-28 19:30:30.603023
29	7	100	POLICE_INTERCEPT	{"status": "Vehicle Intercepted", "location": "Main St Roadblock"}	2026-07-28 19:30:30.603544
30	7	120	INCIDENT_RESOLVED	{"status": "Suspects Arrested", "safetyCleared": true}	2026-07-28 19:30:30.604087
\.


--
-- TOC entry 5307 (class 0 OID 34023)
-- Dependencies: 240
-- Data for Name: traffic_decisions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.traffic_decisions (id, agent, decision_type, location, severity, recommendation, reason, payload, status, source, outcome, created_at) FROM stdin;
1	TrafficAgent	SIGNAL_OPTIMIZATION	Dadar Junction	SEVERE	Increase GREEN +10 sec	High traffic demand	{"prediction": "SEVERE"}	SIMULATED	DEMONSTRATION	SUCCESS	2026-09-01 20:49:05.094153
2	TrafficAgent	ROUTE_RECOMMENDATION	Wadala Junction	HIGH	Divert via Eastern Freeway	Heavy commercial outflow	{"prediction": "HIGH"}	RECOMMENDED	DEMONSTRATION	SUCCESS	2026-09-01 20:42:05.094153
3	TrafficAgent	SIGNAL_OPTIMIZATION	Sion Circle	HIGH	Extend inbound green duration	Merging bottleneck	{"prediction": "HIGH"}	SIMULATED	DEMONSTRATION	SUCCESS	2026-09-01 20:29:05.094153
4	TrafficAgent	ROUTE_RECOMMENDATION	Bandra Junction	MODERATE	Maintain current route	Standard corporate sector outflow	{}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 20:14:05.094153
5	TrafficAgent	TRAFFIC_ANALYSIS	BKC Entry	SEVERE	Wave Synchronization	Aligning 3 consecutive signals	{"prediction": "SEVERE"}	SIMULATED	DEMONSTRATION	SUCCESS	2026-09-01 19:54:05.094153
6	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at Dadar–Wadala corridor. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at Dadar–Wadala corridor. Rerouting active.", "towTrucks": 2, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 6, "expectedCongestion": "88%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 21:40:46.456613
7	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at Dadar–Wadala corridor. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at Dadar–Wadala corridor. Rerouting active.", "towTrucks": 2, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 6, "expectedCongestion": "88%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 21:40:49.422451
8	TrafficAgent	ROAD_CLOSURE_DIVERSION	Wadala	HIGH	Recommend diversion via Sion–Matunga Link	WHAT: road closure at Wadala. WHY: the configured road section is unavailable. DECISION: divert via Sion–Matunga Link. EXPECTED RESULT: avoid the closed road.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Wadala", "prediction": null, "decisionType": "ROAD_CLOSURE_DIVERSION", "currentCondition": "HIGH", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": "wadala-alt", "name": "Sion–Matunga Link", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": ["Wadala Main Road"]}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 21:41:05.764856
9	TrafficAgent	SIGNAL_OPTIMIZATION	Sion Circle	HIGH	Recommend GREEN +5s / RED -5s	WHAT: HIGH traffic detected at Sion Circle. WHY: Assessment combines HIGH traffic inputs.. DECISION: Recommend GREEN +5s / RED -5s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "f051e67e-a2ce-4d43-a2e2-d48bb87e01b3", "currentRed": 45, "currentGreen": 35, "optimizedRed": 40, "redAdjustment": -5, "optimizedGreen": 40, "greenAdjustment": 5, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Sion Circle", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "HIGH", "horizonMinutes": 15, "predictedState": "HIGH"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "HIGH", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "HIGH"}, "predictedCondition": "HIGH", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-01 21:41:53.88357
10	TrafficAgent	SIGNAL_OPTIMIZATION	Bandra Junction	HIGH	Recommend GREEN +5s / RED -5s	WHAT: HIGH traffic detected at Bandra Junction. WHY: Assessment combines HIGH traffic inputs.. DECISION: Recommend GREEN +5s / RED -5s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "17a7ee5c-ab37-4fe4-b99e-013c4cea85d2", "currentRed": 45, "currentGreen": 45, "optimizedRed": 40, "redAdjustment": -5, "optimizedGreen": 50, "greenAdjustment": 5, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Bandra Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "HIGH", "horizonMinutes": 15, "predictedState": "HIGH"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "HIGH", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "HIGH"}, "predictedCondition": "HIGH", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-01 21:42:01.816954
11	TrafficAgent	SIGNAL_OPTIMIZATION	Dadar Junction	SEVERE	Recommend GREEN +8s / RED -8s	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE traffic inputs.. DECISION: Recommend GREEN +8s / RED -8s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "af67a23b-986a-4372-a405-eaecf1f88565", "currentRed": 40, "currentGreen": 30, "optimizedRed": 32, "redAdjustment": -8, "optimizedGreen": 38, "greenAdjustment": 8, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "SEVERE"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "SEVERE", "horizonMinutes": 20, "predictedState": "SEVERE"}, "predictedCondition": "SEVERE", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-01 21:42:03.830791
12	TrafficAgent	SIGNAL_OPTIMIZATION	Worli Junction	MODERATE	Maintain current signal timing	WHAT: MODERATE traffic detected at Worli Junction. WHY: Assessment combines MODERATE traffic inputs.. DECISION: Maintain current signal timing. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "894ff76c-a263-4ed0-9c00-5a3060440eba", "currentRed": 35, "currentGreen": 35, "optimizedRed": 35, "redAdjustment": 0, "optimizedGreen": 35, "greenAdjustment": 0, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Worli Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "MODERATE", "horizonMinutes": 15, "predictedState": "MODERATE"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "MODERATE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "MODERATE", "horizonMinutes": 20, "predictedState": "MODERATE"}, "predictedCondition": "MODERATE", "routeRecommendation": null}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 21:42:09.079211
13	TrafficAgent	SIGNAL_OPTIMIZATION	Dadar Junction	SEVERE	Recommend GREEN +8s / RED -8s	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, SEVERE, SEVERE traffic inputs and the previous observed state.. DECISION: Recommend GREEN +8s / RED -8s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "DADAR-WADALA-01", "currentRed": 48, "currentGreen": 42, "optimizedRed": 40, "redAdjustment": -8, "optimizedGreen": 50, "greenAdjustment": 8, "affectedApproach": "Dadar → Wadala"}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": {"type": "CONGESTION", "reason": "Provided by the configured prediction input.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": {"id": "dadar-matunga-link", "name": "Matunga Link Road", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	SIMULATED	DEMONSTRATION	PENDING	2026-09-01 22:52:41.10847
14	TrafficAgent	EMERGENCY_ROUTE_SIMULATION	Dadar → Sion → KEM Hospital	CRITICAL	Recommend simulated signal priority along the emergency route	SIMULATION ONLY: priority timing is recommended across Dadar Junction, Sion Circle, KEM Hospital approach; no real Mumbai traffic signal is controlled.	{"source": "DEMONSTRATION", "outcome": "PENDING", "location": "Dadar → Sion → KEM Hospital", "simulation": true, "decisionType": "EMERGENCY_ROUTE_SIMULATION", "intersections": ["Dadar Junction", "Sion Circle", "KEM Hospital approach"]}	SIMULATED	DEMONSTRATION	PENDING	2026-09-01 22:57:16.102369
15	TrafficAgent	ROAD_CLOSURE_DIVERSION	{"city":"Mumbai","name":"Wadala","latitude":19.0176,"longitude":72.858}	HIGH	Restrict closed road and monitor diversions	WHAT: road closure at [object Object]. WHY: the configured road section is unavailable. DECISION: restrict the affected approach. EXPECTED RESULT: avoid the closed road.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Wadala", "latitude": 19.0176, "longitude": 72.858}, "prediction": null, "decisionType": "ROAD_CLOSURE_DIVERSION", "currentCondition": "HIGH", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": null}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:04:37.62578
16	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.", "towTrucks": 1, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 2, "expectedCongestion": "42%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:17:06.528673
17	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.", "towTrucks": 1, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 2, "expectedCongestion": "42%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:18:40.483151
18	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.", "towTrucks": 1, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 2, "expectedCongestion": "42%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:19:31.363554
19	TrafficAgent	EMERGENCY_ROUTE_SIMULATION	{"city":"Mumbai","name":"Dadar → Sion → KEM Hospital"}	CRITICAL	Recommend simulated signal priority along the emergency route	SIMULATION ONLY: priority timing is recommended across [object Object]; no real Mumbai traffic signal is controlled.	{"source": "DEMONSTRATION", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar → Sion → KEM Hospital"}, "simulation": true, "decisionType": "EMERGENCY_ROUTE_SIMULATION", "intersections": []}	SIMULATED	DEMONSTRATION	PENDING	2026-09-01 23:19:54.354596
20	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.", "towTrucks": 1, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 2, "expectedCongestion": "42%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:22:56.81364
21	TrafficAgent	ROAD_CLOSURE_DIVERSION	{"city":"Mumbai","name":"Wadala","latitude":19.0176,"longitude":72.858}	HIGH	Restrict closed road and monitor diversions	WHAT: road closure at [object Object]. WHY: the configured road section is unavailable. DECISION: restrict the affected approach. EXPECTED RESULT: avoid the closed road.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Wadala", "latitude": 19.0176, "longitude": 72.858}, "prediction": null, "decisionType": "ROAD_CLOSURE_DIVERSION", "currentCondition": "HIGH", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": null}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:33:54.751962
22	TrafficAgent	ROUTE_RECOMMENDATION	Dadar Junction	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:37:19.105871
23	TrafficAgent	ROUTE_RECOMMENDATION	Dadar Junction	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:42:12.588462
24	TrafficAgent	ROUTE_RECOMMENDATION	Dadar–Wadala corridor	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Dadar–Wadala corridor", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:42:17.819805
25	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.", "towTrucks": 1, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 2, "expectedCongestion": "42%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:42:32.638041
26	TrafficAgent	ROUTE_RECOMMENDATION	Dadar Junction	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-01 23:44:31.687791
27	TrafficAgent	TRAFFIC_ANALYSIS	{"city":"Mumbai","name":"Dadar Junction","latitude":19.0189,"longitude":72.8437}	SEVERE	Monitor traffic condition	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, LOW, SEVERE traffic inputs.. DECISION: Monitor traffic condition. EXPECTED RESULT: Maintain the current signal plan while monitoring for a material change.	{"signal": {}, "source": "DEMONSTRATION", "status": "INSUFFICIENT_DATA", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "prediction": {"type": "CONGESTION", "reason": "Sustained demand may increase congestion based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "TRAFFIC_ANALYSIS", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": null}	INSUFFICIENT_DATA	DEMONSTRATION	PENDING	2026-09-01 23:44:46.167853
28	TrafficAgent	SIGNAL_OPTIMIZATION	Dadar Junction	SEVERE	Recommend GREEN +8s / RED -8s	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE traffic inputs.. DECISION: Recommend GREEN +8s / RED -8s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "af67a23b-986a-4372-a405-eaecf1f88565", "currentRed": 40, "currentGreen": 30, "optimizedRed": 32, "redAdjustment": -8, "optimizedGreen": 38, "greenAdjustment": 8, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "SEVERE"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "SEVERE", "horizonMinutes": 20, "predictedState": "SEVERE"}, "predictedCondition": "SEVERE", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-02 02:00:30.62778
29	TrafficAgent	SIGNAL_OPTIMIZATION	Wadala Junction	HIGH	Recommend GREEN +5s / RED -5s	WHAT: HIGH traffic detected at Wadala Junction. WHY: Assessment combines HIGH traffic inputs.. DECISION: Recommend GREEN +5s / RED -5s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "978f1968-9211-4173-8c3f-02e5a7ef9742", "currentRed": 35, "currentGreen": 35, "optimizedRed": 30, "redAdjustment": -5, "optimizedGreen": 40, "greenAdjustment": 5, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Wadala Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "HIGH", "horizonMinutes": 15, "predictedState": "HIGH"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "HIGH", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "HIGH"}, "predictedCondition": "HIGH", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-02 02:00:34.127344
30	TrafficAgent	SIGNAL_OPTIMIZATION	Sion Circle	HIGH	Recommend GREEN +5s / RED -5s	WHAT: HIGH traffic detected at Sion Circle. WHY: Assessment combines HIGH traffic inputs.. DECISION: Recommend GREEN +5s / RED -5s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "f051e67e-a2ce-4d43-a2e2-d48bb87e01b3", "currentRed": 45, "currentGreen": 35, "optimizedRed": 40, "redAdjustment": -5, "optimizedGreen": 40, "greenAdjustment": 5, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Sion Circle", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "HIGH", "horizonMinutes": 15, "predictedState": "HIGH"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "HIGH", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "HIGH"}, "predictedCondition": "HIGH", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-02 02:00:36.096236
31	TrafficAgent	SIGNAL_OPTIMIZATION	Worli Junction	MODERATE	Maintain current signal timing	WHAT: MODERATE traffic detected at Worli Junction. WHY: Assessment combines MODERATE traffic inputs.. DECISION: Maintain current signal timing. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "894ff76c-a263-4ed0-9c00-5a3060440eba", "currentRed": 35, "currentGreen": 35, "optimizedRed": 35, "redAdjustment": 0, "optimizedGreen": 35, "greenAdjustment": 0, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Worli Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "MODERATE", "horizonMinutes": 15, "predictedState": "MODERATE"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "MODERATE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "MODERATE", "horizonMinutes": 20, "predictedState": "MODERATE"}, "predictedCondition": "MODERATE", "routeRecommendation": null}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-02 02:00:39.303543
32	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.", "towTrucks": 1, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 2, "expectedCongestion": "42%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-02 02:00:57.622034
33	TrafficAgent	TRAFFIC_ANALYSIS	{"city":"Mumbai","name":"Dadar Junction","latitude":19.0189,"longitude":72.8437}	SEVERE	Monitor traffic condition	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, LOW, SEVERE traffic inputs.. DECISION: Monitor traffic condition. EXPECTED RESULT: Maintain the current signal plan while monitoring for a material change.	{"signal": {}, "source": "DEMONSTRATION", "status": "INSUFFICIENT_DATA", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "prediction": {"type": "CONGESTION", "reason": "Sustained demand may increase congestion based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "TRAFFIC_ANALYSIS", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": null}	INSUFFICIENT_DATA	DEMONSTRATION	PENDING	2026-09-02 02:14:34.423911
34	TrafficAgent	TRAFFIC_ANALYSIS	{"city":"Mumbai","name":"Dadar Junction","latitude":19.0189,"longitude":72.8437}	SEVERE	Monitor traffic condition	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, LOW, SEVERE traffic inputs.. DECISION: Monitor traffic condition. EXPECTED RESULT: Maintain the current signal plan while monitoring for a material change.	{"signal": {}, "source": "DEMONSTRATION", "status": "INSUFFICIENT_DATA", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "prediction": {"type": "CONGESTION", "reason": "Sustained demand may increase congestion based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "TRAFFIC_ANALYSIS", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": null}	INSUFFICIENT_DATA	DEMONSTRATION	PENDING	2026-09-05 22:50:04.563067
35	TrafficAgent	TRAFFIC_ANALYSIS	{"city":"Mumbai","name":"Dadar Junction","latitude":19.0189,"longitude":72.8437}	SEVERE	Monitor traffic condition	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, LOW, SEVERE traffic inputs.. DECISION: Monitor traffic condition. EXPECTED RESULT: Maintain the current signal plan while monitoring for a material change.	{"signal": {}, "source": "DEMONSTRATION", "status": "INSUFFICIENT_DATA", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "prediction": {"type": "CONGESTION", "reason": "Sustained demand may increase congestion based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "TRAFFIC_ANALYSIS", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": null}	INSUFFICIENT_DATA	DEMONSTRATION	PENDING	2026-09-08 01:23:07.076348
36	TrafficAgent	TRAFFIC_ANALYSIS	{"city":"Mumbai","name":"Dadar Junction","latitude":19.0189,"longitude":72.8437}	SEVERE	Monitor traffic condition	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, LOW, SEVERE traffic inputs.. DECISION: Monitor traffic condition. EXPECTED RESULT: Maintain the current signal plan while monitoring for a material change.	{"signal": {}, "source": "DEMONSTRATION", "status": "INSUFFICIENT_DATA", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "prediction": {"type": "CONGESTION", "reason": "Sustained demand may increase congestion based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "TRAFFIC_ANALYSIS", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": null}	INSUFFICIENT_DATA	DEMONSTRATION	PENDING	2026-09-08 01:24:50.632937
37	TrafficAgent	TRAFFIC_ANALYSIS	{"city":"Mumbai","name":"Dadar Junction","latitude":19.0189,"longitude":72.8437}	SEVERE	Monitor traffic condition	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, LOW, SEVERE traffic inputs.. DECISION: Monitor traffic condition. EXPECTED RESULT: Maintain the current signal plan while monitoring for a material change.	{"signal": {}, "source": "DEMONSTRATION", "status": "INSUFFICIENT_DATA", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "prediction": {"type": "CONGESTION", "reason": "Sustained demand may increase congestion based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "TRAFFIC_ANALYSIS", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": null}	INSUFFICIENT_DATA	DEMONSTRATION	PENDING	2026-09-08 01:30:14.59308
38	TrafficAgent	ROUTE_RECOMMENDATION	Dadar Junction	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 01:54:08.309654
39	TrafficAgent	TRAFFIC_ANALYSIS	{"city":"Mumbai","name":"Dadar Junction","latitude":19.0189,"longitude":72.8437}	SEVERE	Monitor traffic condition	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, LOW, SEVERE traffic inputs.. DECISION: Monitor traffic condition. EXPECTED RESULT: Maintain the current signal plan while monitoring for a material change.	{"signal": {}, "source": "DEMONSTRATION", "status": "INSUFFICIENT_DATA", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "prediction": {"type": "CONGESTION", "reason": "Sustained demand may increase congestion based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "TRAFFIC_ANALYSIS", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": null}	INSUFFICIENT_DATA	DEMONSTRATION	PENDING	2026-09-08 01:54:16.496969
40	TrafficAgent	ROUTE_RECOMMENDATION	Wadala	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Wadala", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 01:54:17.363823
41	TrafficAgent	ROUTE_RECOMMENDATION	Wadala	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Wadala", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 01:54:25.384939
42	TrafficAgent	ROAD_CLOSURE_DIVERSION	{"city":"Mumbai","name":"Wadala","latitude":19.0176,"longitude":72.858}	HIGH	Restrict closed road and monitor diversions	WHAT: road closure at [object Object]. WHY: the configured road section is unavailable. DECISION: restrict the affected approach. EXPECTED RESULT: avoid the closed road.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Wadala", "latitude": 19.0176, "longitude": 72.858}, "prediction": null, "decisionType": "ROAD_CLOSURE_DIVERSION", "currentCondition": "HIGH", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": null}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 01:54:32.616221
43	TrafficAgent	ROUTE_RECOMMENDATION	Wadala	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Wadala", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 01:54:33.504974
44	TrafficAgent	ROAD_CLOSURE_DIVERSION	{"city":"Mumbai","name":"Wadala","latitude":19.0176,"longitude":72.858}	HIGH	Restrict closed road and monitor diversions	WHAT: road closure at [object Object]. WHY: the configured road section is unavailable. DECISION: restrict the affected approach. EXPECTED RESULT: avoid the closed road.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Wadala", "latitude": 19.0176, "longitude": 72.858}, "prediction": null, "decisionType": "ROAD_CLOSURE_DIVERSION", "currentCondition": "HIGH", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": null}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 01:56:10.304069
45	TrafficAgent	ROAD_CLOSURE_DIVERSION	{"city":"Mumbai","name":"Wadala","latitude":19.0176,"longitude":72.858}	HIGH	Restrict closed road and monitor diversions	WHAT: road closure at [object Object]. WHY: the configured road section is unavailable. DECISION: restrict the affected approach. EXPECTED RESULT: avoid the closed road.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Wadala", "latitude": 19.0176, "longitude": 72.858}, "prediction": null, "decisionType": "ROAD_CLOSURE_DIVERSION", "currentCondition": "HIGH", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": null}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 01:56:39.495813
46	TrafficAgent	ROAD_CLOSURE_DIVERSION	{"city":"Mumbai","name":"Wadala","latitude":19.0176,"longitude":72.858}	HIGH	Restrict closed road and monitor diversions	WHAT: road closure at [object Object]. WHY: the configured road section is unavailable. DECISION: restrict the affected approach. EXPECTED RESULT: avoid the closed road.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Wadala", "latitude": 19.0176, "longitude": 72.858}, "prediction": null, "decisionType": "ROAD_CLOSURE_DIVERSION", "currentCondition": "HIGH", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": null}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 01:56:57.052635
47	TrafficAgent	ROUTE_RECOMMENDATION	Wadala	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Wadala", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 01:59:59.597551
48	TrafficAgent	ROUTE_RECOMMENDATION	Wadala	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Wadala", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 02:00:21.15971
49	TrafficAgent	ROUTE_RECOMMENDATION	Dadar Junction	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 02:00:21.235201
50	TrafficAgent	ROUTE_RECOMMENDATION	Dadar Junction	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION DATA", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION DATA	PENDING	2026-09-08 02:07:18.377752
51	TrafficAgent	SIGNAL_OPTIMIZATION	Dadar Junction	SEVERE	Recommend GREEN +8s / RED -8s	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE traffic inputs.. DECISION: Recommend GREEN +8s / RED -8s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "af67a23b-986a-4372-a405-eaecf1f88565", "currentRed": 40, "currentGreen": 30, "optimizedRed": 32, "redAdjustment": -8, "optimizedGreen": 38, "greenAdjustment": 8, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "SEVERE"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "SEVERE", "horizonMinutes": 20, "predictedState": "SEVERE"}, "predictedCondition": "SEVERE", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-08 13:10:38.353198
52	TrafficAgent	SIGNAL_OPTIMIZATION	Wadala Junction	SEVERE	Recommend GREEN +8s / RED -8s	WHAT: SEVERE traffic detected at Wadala Junction. WHY: Assessment combines SEVERE traffic inputs.. DECISION: Recommend GREEN +8s / RED -8s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "978f1968-9211-4173-8c3f-02e5a7ef9742", "currentRed": 35, "currentGreen": 35, "optimizedRed": 27, "redAdjustment": -8, "optimizedGreen": 43, "greenAdjustment": 8, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Wadala Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "SEVERE"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "SEVERE", "horizonMinutes": 20, "predictedState": "SEVERE"}, "predictedCondition": "SEVERE", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-08 13:10:40.604383
53	TrafficAgent	SIGNAL_OPTIMIZATION	Sion Circle	SEVERE	Recommend GREEN +8s / RED -8s	WHAT: SEVERE traffic detected at Sion Circle. WHY: Assessment combines SEVERE traffic inputs.. DECISION: Recommend GREEN +8s / RED -8s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "f051e67e-a2ce-4d43-a2e2-d48bb87e01b3", "currentRed": 45, "currentGreen": 35, "optimizedRed": 37, "redAdjustment": -8, "optimizedGreen": 43, "greenAdjustment": 8, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Sion Circle", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "SEVERE"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "SEVERE", "horizonMinutes": 20, "predictedState": "SEVERE"}, "predictedCondition": "SEVERE", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-08 13:10:42.771937
54	TrafficAgent	SIGNAL_OPTIMIZATION	Bandra Junction	SEVERE	Recommend GREEN +8s / RED -8s	WHAT: SEVERE traffic detected at Bandra Junction. WHY: Assessment combines SEVERE traffic inputs.. DECISION: Recommend GREEN +8s / RED -8s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "17a7ee5c-ab37-4fe4-b99e-013c4cea85d2", "currentRed": 45, "currentGreen": 45, "optimizedRed": 37, "redAdjustment": -8, "optimizedGreen": 53, "greenAdjustment": 8, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Bandra Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "SEVERE"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "SEVERE", "horizonMinutes": 20, "predictedState": "SEVERE"}, "predictedCondition": "SEVERE", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-08 13:10:46.428577
55	TrafficAgent	SIGNAL_OPTIMIZATION	Matunga Junction	HIGH	Recommend GREEN +5s / RED -5s	WHAT: HIGH traffic detected at Matunga Junction. WHY: Assessment combines HIGH traffic inputs.. DECISION: Recommend GREEN +5s / RED -5s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "786fd97f-5099-41b5-b5c9-d1fc89d1ed4e", "currentRed": 30, "currentGreen": 30, "optimizedRed": 25, "redAdjustment": -5, "optimizedGreen": 35, "greenAdjustment": 5, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Matunga Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "HIGH", "horizonMinutes": 15, "predictedState": "HIGH"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "HIGH", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "HIGH"}, "predictedCondition": "HIGH", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-08 13:10:47.491947
56	TrafficAgent	SIGNAL_OPTIMIZATION	Worli Junction	HIGH	Recommend GREEN +5s / RED -5s	WHAT: HIGH traffic detected at Worli Junction. WHY: Assessment combines HIGH traffic inputs.. DECISION: Recommend GREEN +5s / RED -5s. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "894ff76c-a263-4ed0-9c00-5a3060440eba", "currentRed": 35, "currentGreen": 35, "optimizedRed": 30, "redAdjustment": -5, "optimizedGreen": 40, "greenAdjustment": 5, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "SIMULATED", "outcome": "PENDING", "location": "Worli Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "HIGH", "horizonMinutes": 15, "predictedState": "HIGH"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "HIGH", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "HIGH"}, "predictedCondition": "HIGH", "routeRecommendation": null}	SIMULATED	DEMONSTRATION	PENDING	2026-09-08 13:10:49.052049
57	TrafficAgent	SIGNAL_OPTIMIZATION	BKC Junction	MODERATE	Maintain current signal timing	WHAT: MODERATE traffic detected at BKC Junction. WHY: Assessment combines MODERATE traffic inputs.. DECISION: Maintain current signal timing. EXPECTED RESULT: Reduce queue buildup on the affected approach while preserving the configured signal cycle.	{"signal": {"signalId": "9ea42355-89ca-46f0-b58b-6a37714d64fd", "currentRed": 20, "currentGreen": 40, "optimizedRed": 20, "redAdjustment": 0, "optimizedGreen": 40, "greenAdjustment": 0, "affectedApproach": null}, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "BKC Junction", "prediction": {"type": "CONGESTION", "reason": "Current conditions are expected to persist based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "MODERATE", "horizonMinutes": 15, "predictedState": "MODERATE"}, "decisionType": "SIGNAL_OPTIMIZATION", "currentCondition": "MODERATE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "MODERATE", "horizonMinutes": 20, "predictedState": "MODERATE"}, "predictedCondition": "MODERATE", "routeRecommendation": null}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 13:10:52.188714
58	TrafficAgent	TRAFFIC_ANALYSIS	{"city":"Mumbai","name":"Dadar Junction","latitude":19.0189,"longitude":72.8437}	SEVERE	Monitor traffic condition	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, LOW, SEVERE traffic inputs.. DECISION: Monitor traffic condition. EXPECTED RESULT: Maintain the current signal plan while monitoring for a material change.	{"signal": {}, "source": "DEMONSTRATION", "status": "INSUFFICIENT_DATA", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "prediction": {"type": "CONGESTION", "reason": "Sustained demand may increase congestion based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "TRAFFIC_ANALYSIS", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": null}	INSUFFICIENT_DATA	DEMONSTRATION	PENDING	2026-09-08 13:19:38.346145
59	TrafficAgent	ROUTE_RECOMMENDATION	Dadar Junction	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Dadar Junction", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 13:19:49.470901
60	TrafficAgent	ROUTE_RECOMMENDATION	Dadar Station, Mumbai	Moderate	Recommend Route A	WHAT: Route A was evaluated against available route conditions. WHY: Lower predicted traffic impact from the supplied route conditions.. DECISION: Recommend Route A. EXPECTED RESULT: Lower predicted traffic impact for the journey.	{"signal": null, "source": "DEMONSTRATION", "status": "RECOMMENDED", "outcome": "PENDING", "location": "Dadar Station, Mumbai", "prediction": null, "decisionType": "ROUTE_RECOMMENDATION", "currentCondition": "MODERATE", "densityPrediction": null, "predictedCondition": null, "routeRecommendation": {"id": 0, "name": "Route A", "reason": "Lower predicted traffic impact from the supplied route conditions.", "source": "DEMONSTRATION", "affectedRoutes": []}}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-08 13:28:26.288465
61	TrafficAgent	TRAFFIC_ANALYSIS	{"city":"Mumbai","name":"Dadar Junction","latitude":19.0189,"longitude":72.8437}	SEVERE	Monitor traffic condition	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, LOW, SEVERE traffic inputs.. DECISION: Monitor traffic condition. EXPECTED RESULT: Maintain the current signal plan while monitoring for a material change.	{"signal": {}, "source": "DEMONSTRATION", "status": "INSUFFICIENT_DATA", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "prediction": {"type": "CONGESTION", "reason": "Sustained demand may increase congestion based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "TRAFFIC_ANALYSIS", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": null}	INSUFFICIENT_DATA	DEMONSTRATION	PENDING	2026-09-08 18:00:54.932199
62	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.", "towTrucks": 1, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 2, "expectedCongestion": "42%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-30 21:46:12.428124
63	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.", "towTrucks": 1, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 2, "expectedCongestion": "42%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-30 22:37:17.436854
64	TrafficAgent	TRAFFIC_ANALYSIS	\N	HIGH	Close affected road and initiate diversion	Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.	{"risks": ["Secondary collisions", "Gridlock on alternative routes"], "reasoning": "Collision involving heavy vehicles. Lanes blocked at [object Object]. Rerouting active.", "towTrucks": 1, "confidence": "92%", "collaborators": ["PoliceAgent", "HospitalAgent"], "officersDeployed": 2, "expectedCongestion": "42%"}	RECOMMENDED	DEMONSTRATION	PENDING	2026-09-30 22:37:36.434851
65	TrafficAgent	TRAFFIC_ANALYSIS	{"city":"Mumbai","name":"Dadar Junction","latitude":19.0189,"longitude":72.8437}	SEVERE	Monitor traffic condition	WHAT: SEVERE traffic detected at Dadar Junction. WHY: Assessment combines SEVERE, HIGH, HIGH, LOW, SEVERE traffic inputs.. DECISION: Monitor traffic condition. EXPECTED RESULT: Maintain the current signal plan while monitoring for a material change.	{"signal": {}, "source": "DEMONSTRATION", "status": "INSUFFICIENT_DATA", "outcome": "PENDING", "location": {"city": "Mumbai", "name": "Dadar Junction", "latitude": 19.0189, "longitude": 72.8437}, "prediction": {"type": "CONGESTION", "reason": "Sustained demand may increase congestion based on deterministic demonstration rules.", "source": "DEMONSTRATION", "confidence": 0.8, "currentState": "SEVERE", "horizonMinutes": 15, "predictedState": "CRITICAL"}, "decisionType": "TRAFFIC_ANALYSIS", "currentCondition": "SEVERE", "densityPrediction": {"type": "DENSITY", "reason": "Deterministic demonstration estimate based on density and signal demand.", "source": "DEMONSTRATION", "confidence": 0.75, "currentState": "HIGH", "horizonMinutes": 20, "predictedState": "CRITICAL"}, "predictedCondition": "CRITICAL", "routeRecommendation": null}	INSUFFICIENT_DATA	DEMONSTRATION	PENDING	2026-10-02 15:48:01.968417
\.


--
-- TOC entry 5309 (class 0 OID 34041)
-- Dependencies: 242
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
-- TOC entry 5311 (class 0 OID 34078)
-- Dependencies: 244
-- Data for Name: traffic_signal_monitoring; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.traffic_signal_monitoring (id, entity_id, signal_code, intersection_name, area, current_phase, current_green_seconds, current_red_seconds, cycle_time_seconds, traffic_level, density_level, affected_approach, predicted_traffic_level, predicted_density_level, prediction_horizon_minutes, ai_decision, ai_recommendation, recommended_green_seconds, recommended_red_seconds, green_adjustment_seconds, red_adjustment_seconds, expected_impact, status, reason, confidence, last_observed_at, updated_at) FROM stdin;
36	b50d7246-f7f4-4e73-a13a-373c4faf2a3b	SIG-BKC-001	BKC Kalanagar Junction	BKC	RED	40	30	70	LOW	LOW	BKC - Kalanagar / Kurla approach	LOW	LOW	20	Maintain Timing	Maintain current signal timing	40	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	83.20	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
3	f051e67e-a2ce-4d43-a2e2-d48bb87e01b3	SIG-f051e6	Sion Circle	Sion	RED	50	35	85	SEVERE	VERY_HIGH	Sion Circle - Eastern Express Highway approach	CRITICAL	VERY_HIGH	10	Increase Green	Increase green phase by 8s; reduce red phase accordingly	58	27	8	-8	Reduce inbound queue accumulation	OPTIMIZATION RECOMMENDED	High inbound demand and sustained queue accumulation detected. Traffic Agent recommends additional green time for the affected approach.	93.10	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
1	af67a23b-986a-4372-a405-eaecf1f88565	SIG-af67a2	Dadar Junction	Dadar	GREEN	40	35	75	MODERATE	LOW	Shivaji Park - Dadar approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	84.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
2	978f1968-9211-4173-8c3f-02e5a7ef9742	SIG-978f19	Wadala Junction	Wadala	RED	40	35	75	MODERATE	LOW	Wadala - Sion / Dadar approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	85.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
4	786fd97f-5099-41b5-b5c9-d1fc89d1ed4e	SIG-786fd9	Matunga Junction	Matunga	GREEN	55	35	90	SEVERE	HIGH	Matunga - Five Gardens / Sion approach	SEVERE	HIGH	10	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	60	30	5	-5	Improve progression between adjacent intersections	OPTIMIZATION RECOMMENDED	Severe traffic conditions detected across the corridor. Traffic Agent recommends coordinating the next cycle with adjacent intersections.	93.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
5	17a7ee5c-ab37-4fe4-b99e-013c4cea85d2	SIG-17a7ee	Bandra Junction	Bandra	RED	45	35	80	HIGH	MODERATE	Bandra - Western Express Highway approach	HIGH	MODERATE	15	Increase Green	Increase green phase by 8s; reduce red phase accordingly	53	27	8	-8	Improve traffic flow through the affected approach	RECOMMENDATION READY	Traffic demand is above normal operating levels. Additional green time is recommended to improve vehicle discharge.	91.50	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
6	9ea42355-89ca-46f0-b58b-6a37714d64fd	SIG-9ea423	BKC Junction	BKC	RED	40	35	75	MODERATE	LOW	BKC - Kalanagar / Kurla approach	MODERATE	MODERATE	20	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	45	30	5	-5	Improve progression between adjacent intersections	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	88.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
7	894ff76c-a263-4ed0-9c00-5a3060440eba	SIG-894ff7	Worli Junction	Worli	GREEN	40	35	75	MODERATE	LOW	Worli - Annie Besant Road approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	89.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
14	fde330ce-138f-4072-ae10-051435acfedc	SIG-SOU-007	Mahalaxmi Junction	Mahalaxmi	RED	35	30	65	LOW	LOW	Mahalaxmi - Dr E Moses Road approach	LOW	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	84.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
15	536ccf71-174d-451d-ad0c-d6143c692292	SIG-SOU-008	Tardeo Junction	Tardeo	RED	35	30	65	LOW	LOW	Tardeo - Nana Chowk approach	MODERATE	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	85.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
16	0773e544-c4b3-405f-afdc-e76f70f8af09	SIG-SOU-009	Mumbai Central Junction	Mumbai Central	GREEN	50	35	85	HIGH	MODERATE	Mumbai Central - Bellasis Road approach	SEVERE	HIGH	15	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	55	30	5	-5	Improve progression between adjacent intersections	RECOMMENDATION READY	Elevated traffic demand detected. Traffic Agent recommends corridor coordination to improve progression.	89.40	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
17	0eb75675-e91f-4484-90b4-f0bc2cbb6cd7	SIG-SOU-010	Byculla Junction	Byculla	RED	40	35	75	MODERATE	LOW	Byculla - Dr Babasaheb Ambedkar Road approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	84.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
18	bc276040-2430-41fb-b1ba-2cecf867a4e2	SIG-WOR-001	Worli Naka Junction	Worli	RED	40	35	75	MODERATE	LOW	Worli - Annie Besant Road approach	MODERATE	MODERATE	20	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	45	30	5	-5	Improve progression between adjacent intersections	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	85.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
19	19ddd422-42ce-4da2-bace-7a3335b3b4cc	SIG-WOR-002	Worli Sea Face Junction	Worli	GREEN	40	35	75	MODERATE	LOW	Worli - Annie Besant Road approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	86.40	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
20	96e5b191-b4f4-4dd1-9843-ae18b746fc19	SIG-WOR-003	Prabhadevi Junction	Prabhadevi	RED	45	35	80	MODERATE	MODERATE	Prabhadevi - Gokhale Road approach	HIGH	HIGH	20	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	87.20	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
21	8e9bff0f-f9d6-4318-9030-555a7c097969	SIG-WOR-004	Lower Parel Junction	Lower Parel	RED	50	35	85	SEVERE	VERY_HIGH	Lower Parel - Senapati Bapat Marg approach	CRITICAL	VERY_HIGH	10	Increase Green	Increase green phase by 8s; reduce red phase accordingly	58	27	8	-8	Reduce inbound queue accumulation	OPTIMIZATION RECOMMENDED	High inbound demand and sustained queue accumulation detected. Traffic Agent recommends additional green time for the affected approach.	93.10	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
22	c6cfb255-4dd2-4223-a1c3-79170c77ecf5	SIG-WOR-005	Elphinstone Junction	Elphinstone	GREEN	40	35	75	MODERATE	LOW	Elphinstone - Senapati Bapat Marg approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	88.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
23	33638101-04ca-4bed-839f-7702216108ba	SIG-DAD-001	Dadar TT Circle	Dadar	RED	40	35	75	MODERATE	LOW	Shivaji Park - Dadar approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	89.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
24	a7d603ae-72c4-4905-b0db-e012d89a2e46	SIG-DAD-002	Dadar Shivaji Park Junction	Dadar	RED	50	35	85	HIGH	HIGH	Dadar TT - Wadala approach	SEVERE	HIGH	15	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	55	30	5	-5	Improve progression between adjacent intersections	RECOMMENDATION READY	Elevated traffic demand detected. Traffic Agent recommends corridor coordination to improve progression.	90.10	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
25	c5980fde-e36e-4c20-b660-febb435cbc54	SIG-MAH-001	Mahim Junction	Mahim	GREEN	40	35	75	MODERATE	LOW	Mahim - S.V. Road / L.J. Road approach	HIGH	HIGH	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	84.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
26	37c451e1-2bcf-4202-8d16-7b33fe9b8abc	SIG-MAH-002	Mahim Causeway Junction	Mahim	RED	45	35	80	HIGH	MODERATE	Mahim - S.V. Road / L.J. Road approach	MODERATE	MODERATE	15	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic volume is elevated but current signal timing remains within acceptable operating conditions. Continued monitoring is recommended.	91.50	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
27	d803906e-7f3e-4276-847a-2c1137d7b38c	SIG-MAT-001	Matunga Five Gardens Junction	Matunga	RED	35	30	65	LOW	LOW	Matunga - Five Gardens / Sion approach	LOW	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	82.40	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
28	39106148-7a6b-4be2-8941-a1c0f8b38338	SIG-SIO-001	Sion Circle	Sion	GREEN	45	35	80	MODERATE	MODERATE	Sion Circle - Eastern Express Highway approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	87.20	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
29	cddef124-03d6-4080-ba9e-5d0d13186e04	SIG-SIO-002	Sion Hospital Junction	Sion	RED	40	35	75	MODERATE	LOW	Sion Circle - Eastern Express Highway approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	88.00	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
30	cd8353e5-35f3-4908-8bd8-7469bea036f7	SIG-BAN-001	Bandra Junction	Bandra	RED	40	35	75	MODERATE	LOW	Bandra - Western Express Highway approach	HIGH	HIGH	20	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	45	30	5	-5	Improve progression between adjacent intersections	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	88.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
31	d917ad96-6f35-4d74-88f3-145bce83bcd1	SIG-BAN-002	Bandra Bandstand Junction	Bandra	GREEN	45	35	80	HIGH	MODERATE	Bandra - Western Express Highway approach	MODERATE	MODERATE	15	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic volume is elevated but current signal timing remains within acceptable operating conditions. Continued monitoring is recommended.	90.10	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
32	0f254671-b2d1-4595-833d-f5c6ccbdca98	SIG-BAN-003	Khar Junction	Khar	RED	50	35	85	HIGH	MODERATE	Primary inbound approach	SEVERE	HIGH	15	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	55	30	5	-5	Improve progression between adjacent intersections	RECOMMENDATION READY	Elevated traffic demand detected. Traffic Agent recommends corridor coordination to improve progression.	90.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
33	ad66ef0b-eb35-438a-a350-e39aef911b67	SIG-SAN-001	Santacruz Junction	Santacruz	RED	45	35	80	HIGH	HIGH	Santacruz - S.V. Road / Western Express Highway approach	HIGH	MODERATE	15	Increase Green	Increase green phase by 8s; reduce red phase accordingly	53	27	8	-8	Improve traffic flow through the affected approach	RECOMMENDATION READY	Traffic demand is above normal operating levels. Additional green time is recommended to improve vehicle discharge.	91.50	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
34	b9bd40cb-44cc-45f7-a00d-2197ce85ec0b	SIG-SAN-002	Vakola Junction	Santacruz	GREEN	40	35	75	MODERATE	LOW	Santacruz - S.V. Road / Western Express Highway approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	85.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
52	7eaa9651-3978-4d0c-93db-af1aa900006b	SIG-KAN-001	Kandivali Junction	Kandivali	GREEN	45	35	80	MODERATE	MODERATE	Kandivali - Western Express Highway approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	87.20	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
35	bb879fcb-3700-41a4-a25a-6fcfe4cce058	SIG-SAN-003	Milan Subway Junction	Santacruz	RED	50	35	85	SEVERE	HIGH	Santacruz - S.V. Road / Western Express Highway approach	SEVERE	HIGH	10	Increase Green	Increase green phase by 8s; reduce red phase accordingly	58	27	8	-8	Reduce inbound queue accumulation	OPTIMIZATION RECOMMENDED	High inbound demand and sustained queue accumulation detected. Traffic Agent recommends additional green time for the affected approach.	94.50	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
37	b40fdc72-96e2-4048-9a69-5086b61fb44a	SIG-BKC-002	BKC Bharat Diamond Bourse Junction	BKC	GREEN	35	30	65	LOW	LOW	BKC - Kalanagar / Kurla approach	LOW	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	84.00	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
38	e49660f6-23f0-4f98-998e-5692e62dba62	SIG-BKC-003	BKC Kurla Junction	BKC	RED	35	30	65	LOW	LOW	BKC - Kalanagar / Kurla approach	LOW	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	84.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
39	28ec1ca1-fc65-45fa-ad45-018c15001636	SIG-AND-001	Andheri Station Junction	Andheri	RED	50	35	85	SEVERE	VERY_HIGH	Andheri - Andheri Kurla Road approach	CRITICAL	VERY_HIGH	10	Increase Green	Increase green phase by 8s; reduce red phase accordingly	58	27	8	-8	Reduce inbound queue accumulation	OPTIMIZATION RECOMMENDED	High inbound demand and sustained queue accumulation detected. Traffic Agent recommends additional green time for the affected approach.	93.10	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
40	b2d15ebb-886b-497f-950a-17cc46ae430a	SIG-AND-002	Andheri Market Junction	Andheri	GREEN	55	35	90	SEVERE	HIGH	Andheri - Andheri Kurla Road approach	SEVERE	HIGH	10	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	60	30	5	-5	Improve progression between adjacent intersections	OPTIMIZATION RECOMMENDED	Severe traffic conditions detected across the corridor. Traffic Agent recommends coordinating the next cycle with adjacent intersections.	93.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
41	5691f2d5-fbe4-4c22-a9c1-3d5ef0a7ebb8	SIG-AND-003	Andheri Kurla Junction	Andheri	RED	40	35	75	MODERATE	LOW	Andheri - Andheri Kurla Road approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	84.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
42	99513606-d20e-4938-9646-417b96a52a80	SIG-AND-004	Chakala Junction	Andheri	RED	35	30	65	LOW	LOW	Andheri - Andheri Kurla Road approach	LOW	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	81.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
43	117791e3-f16d-4c09-bdec-ad2514c31cde	SIG-AND-005	Marol Naka Junction	Marol	GREEN	45	35	80	HIGH	MODERATE	Marol Naka - Andheri Kurla Road approach	MODERATE	MODERATE	15	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic volume is elevated but current signal timing remains within acceptable operating conditions. Continued monitoring is recommended.	88.70	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
44	cb673f52-23d2-43e5-8333-383c518c8ee7	SIG-AND-006	Saki Naka Junction	Saki Naka	RED	50	35	85	HIGH	MODERATE	Saki Naka - Saki Vihar / Andheri Kurla approach	SEVERE	HIGH	15	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	55	30	5	-5	Improve progression between adjacent intersections	RECOMMENDATION READY	Elevated traffic demand detected. Traffic Agent recommends corridor coordination to improve progression.	89.40	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
45	9bfcaf41-1f05-47a7-a55b-c4206b504378	SIG-JOG-001	Jogeshwari Junction	Jogeshwari	RED	50	35	85	SEVERE	VERY_HIGH	Jogeshwari - Western Express Highway approach	CRITICAL	VERY_HIGH	10	Increase Green	Increase green phase by 8s; reduce red phase accordingly	58	27	8	-8	Reduce inbound queue accumulation	OPTIMIZATION RECOMMENDED	High inbound demand and sustained queue accumulation detected. Traffic Agent recommends additional green time for the affected approach.	93.10	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
46	85167a8d-e080-4bb1-ad94-d84e00bc8912	SIG-JOG-002	JVLR Western Junction	Jogeshwari	GREEN	40	35	75	MODERATE	LOW	Jogeshwari - Western Express Highway approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	88.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
47	80b13ccf-8b0a-49a4-9903-9a5f9739037c	SIG-GOR-001	Goregaon Junction	Goregaon	RED	40	35	75	MODERATE	LOW	Goregaon - Western Express Highway approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	89.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
48	19e54f28-8d6c-4a98-96c0-fc14b2e62590	SIG-GOR-002	Goregaon WEH Junction	Goregaon	RED	45	35	80	MODERATE	MODERATE	Goregaon - Western Express Highway approach	MODERATE	MODERATE	20	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	50	30	5	-5	Improve progression between adjacent intersections	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	84.00	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
49	8fc58d1f-e4ef-4e11-8326-5ccae1104aee	SIG-GOR-003	Aarey Junction	Aarey	GREEN	55	30	85	CRITICAL	VERY_HIGH	Aarey - JVLR approach	CRITICAL	VERY_HIGH	10	Increase Green	Increase green phase by 8s; reduce red phase accordingly	63	22	8	-8	Reduce queue growth and improve corridor throughput	OPTIMIZATION RECOMMENDED	Sustained queue buildup and very high vehicle density detected. Traffic Agent recommends extending the inbound green phase to reduce queue growth.	97.20	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
50	1c7d00d0-c315-4bdb-be93-545d408ef92f	SIG-MAL-001	Malad Junction	Malad	RED	35	30	65	LOW	LOW	Malad - Link Road approach	MODERATE	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	81.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
51	b4fa4333-7ccd-4973-8390-fb9c073f7538	SIG-MAL-002	Malad Link Road Junction	Malad	RED	40	35	75	MODERATE	LOW	Malad - Link Road approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	86.40	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
53	a295064f-9b20-4116-a083-a0a6db2af40f	SIG-BOR-001	Borivali Junction	Borivali	RED	50	35	85	SEVERE	HIGH	Borivali - Western Express Highway approach	SEVERE	HIGH	10	Increase Green	Increase green phase by 8s; reduce red phase accordingly	58	27	8	-8	Reduce inbound queue accumulation	OPTIMIZATION RECOMMENDED	High inbound demand and sustained queue accumulation detected. Traffic Agent recommends additional green time for the affected approach.	94.50	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
54	20176dbe-c336-452d-b582-c33bc641e8ea	SIG-BOR-002	Borivali WEH Junction	Borivali	RED	40	35	75	MODERATE	LOW	Borivali - Western Express Highway approach	MODERATE	MODERATE	20	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	45	30	5	-5	Improve progression between adjacent intersections	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	88.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
55	550c3b5d-c034-4f6f-8c11-ad2012618545	SIG-KUR-001	Kurla Depot Junction	Kurla	GREEN	35	30	65	LOW	LOW	Kurla - LBS Marg approach	MODERATE	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	85.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
56	67ca2e85-012b-40da-a559-24dd4f9ebc92	SIG-KUR-002	Kurla Station Junction	Kurla	RED	45	35	80	MODERATE	MODERATE	Kurla - LBS Marg approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	84.00	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
57	7adb3499-e4da-425a-b1be-52de6d8ceeeb	SIG-GHA-001	Ghatkopar Junction	Ghatkopar	RED	45	35	80	HIGH	HIGH	Ghatkopar - LBS Marg approach	HIGH	MODERATE	15	Increase Green	Increase green phase by 8s; reduce red phase accordingly	53	27	8	-8	Improve traffic flow through the affected approach	RECOMMENDATION READY	Traffic demand is above normal operating levels. Additional green time is recommended to improve vehicle discharge.	88.70	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
58	33751247-ce67-46e8-89dd-4dfd9199b295	SIG-GHA-002	Ghatkopar West Junction	Ghatkopar	GREEN	40	35	75	MODERATE	LOW	Ghatkopar - LBS Marg approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	85.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
59	d371eb4a-fc63-4dd7-a75a-fd2012a1f201	SIG-VIK-001	Vikhroli Junction	Vikhroli	RED	45	35	80	HIGH	MODERATE	Vikhroli - Eastern Express Highway approach	MODERATE	MODERATE	15	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic volume is elevated but current signal timing remains within acceptable operating conditions. Continued monitoring is recommended.	90.10	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
70	d2e6c6f9-e4c1-4902-968a-9f27699b3748	SIG-MAN-001	Mankhurd Junction	Mankhurd	GREEN	45	35	80	HIGH	MODERATE	Mankhurd - Sion Panvel Road approach	MODERATE	MODERATE	15	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic volume is elevated but current signal timing remains within acceptable operating conditions. Continued monitoring is recommended.	88.00	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
71	8a580f56-7a2a-4c15-a552-dce5a0e4f7d1	SIG-GOV-001	Govandi Junction	Govandi	RED	45	35	80	HIGH	MODERATE	Govandi - Sion Panvel Road approach	MODERATE	MODERATE	15	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic volume is elevated but current signal timing remains within acceptable operating conditions. Continued monitoring is recommended.	88.70	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
60	6fefba03-216d-48d1-bba6-1c7f89de2615	SIG-VIK-002	Vikhroli Parksite Junction	Vikhroli	RED	50	35	85	HIGH	HIGH	Vikhroli - Eastern Express Highway approach	SEVERE	HIGH	15	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	55	30	5	-5	Improve progression between adjacent intersections	RECOMMENDATION READY	Elevated traffic demand detected. Traffic Agent recommends corridor coordination to improve progression.	90.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
61	8a0f30a0-9057-47d3-8c05-07a252bfd48e	SIG-BHA-001	Bhandup Junction	Bhandup	GREEN	35	30	65	LOW	LOW	Bhandup - LBS Marg approach	LOW	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	84.00	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
62	7ed2754f-d89d-434c-94ff-fb109924b9c7	SIG-MUL-001	Mulund Check Naka	Mulund	RED	50	35	85	SEVERE	HIGH	Mulund - LBS Marg / Eastern Express Highway approach	SEVERE	HIGH	10	Increase Green	Increase green phase by 8s; reduce red phase accordingly	58	27	8	-8	Reduce inbound queue accumulation	OPTIMIZATION RECOMMENDED	High inbound demand and sustained queue accumulation detected. Traffic Agent recommends additional green time for the affected approach.	92.40	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
63	a75b7c65-7524-49bd-9a37-3db44901a55d	SIG-CHE-001	Chembur Junction	Chembur	RED	40	35	75	MODERATE	LOW	Chembur - Sion Panvel Highway approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	89.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
64	65aa004f-a2b2-4eac-84a9-c09fb619a789	SIG-CHE-002	Amar Mahal Junction	Chembur	GREEN	45	35	80	MODERATE	MODERATE	Chembur - Sion Panvel Highway approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	84.00	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
65	ce390c23-f218-4ea4-a68b-2a9caa84a093	SIG-CHE-003	RCF Junction	Chembur	RED	40	35	75	MODERATE	LOW	Chembur - Sion Panvel Highway approach	HIGH	HIGH	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	84.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
66	f29ef3ee-b345-4bf8-9e69-039bb660b725	SIG-CHE-004	Vashi Naka Junction	Chembur	RED	35	30	65	LOW	LOW	Chembur - Sion Panvel Highway approach	LOW	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	81.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
67	f20d118f-085d-4c93-a693-d2fa999ad809	SIG-WAD-001	Wadala Junction	Wadala	GREEN	45	35	80	HIGH	MODERATE	Wadala - Sion / Dadar approach	MODERATE	MODERATE	15	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic volume is elevated but current signal timing remains within acceptable operating conditions. Continued monitoring is recommended.	90.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
68	ac0f6750-b632-459c-ad10-f9c89e684605	SIG-WAD-002	Wadala IMAX Junction	Wadala	RED	45	35	80	MODERATE	MODERATE	Wadala - Sion / Dadar approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	45	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	87.20	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
69	d5345fab-0739-4f93-9193-58afae8af705	SIG-WAD-003	Antop Hill Junction	Antop Hill	RED	35	30	65	LOW	LOW	Primary inbound approach	LOW	LOW	20	Maintain Timing	Maintain current signal timing	35	30	0	0	Maintain stable intersection throughput	MONITORING	Traffic demand is currently low. Existing signal timing is sufficient for current conditions.	84.00	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
8	6195470e-4031-4f89-8857-5ce348d50f04	SIG-SOU-001	Hutatma Chowk Junction	Fort	RED	50	35	85	HIGH	MODERATE	Fort - CSMT / Marine Drive approach	SEVERE	HIGH	15	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	55	30	5	-5	Improve progression between adjacent intersections	RECOMMENDATION READY	Elevated traffic demand detected. Traffic Agent recommends corridor coordination to improve progression.	88.70	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
9	5384a3cd-3839-42b6-bbf4-fb8d3c2fc0f7	SIG-SOU-002	CSMT Junction	Fort	RED	40	35	75	MODERATE	LOW	Fort - CSMT / Marine Drive approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	84.80	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
10	622338b9-b794-46c0-9b95-1ece33df3d8c	SIG-SOU-003	Churchgate Junction	Churchgate	GREEN	40	35	75	MODERATE	LOW	Churchgate - Marine Drive approach	HIGH	HIGH	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	85.60	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
11	5883c94a-fa0f-4a78-9ee0-c5f9c4b273b7	SIG-SOU-004	Marine Drive Junction	Marine Drive	RED	50	35	85	SEVERE	HIGH	Marine Drive - Nariman Point approach	SEVERE	HIGH	10	Increase Green	Increase green phase by 8s; reduce red phase accordingly	58	27	8	-8	Reduce inbound queue accumulation	OPTIMIZATION RECOMMENDED	High inbound demand and sustained queue accumulation detected. Traffic Agent recommends additional green time for the affected approach.	94.50	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
12	4407747f-e360-42c0-bfa6-df2afdc27c2a	SIG-SOU-005	Nariman Point Junction	Nariman Point	RED	50	35	85	HIGH	HIGH	Nariman Point - Marine Drive approach	SEVERE	HIGH	15	Coordinate Corridor	Coordinate next signal cycle with adjacent junctions	55	30	5	-5	Improve progression between adjacent intersections	RECOMMENDATION READY	Elevated traffic demand detected. Traffic Agent recommends corridor coordination to improve progression.	91.50	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
13	ec145b30-c966-49a4-ab9b-3285f6695c02	SIG-SOU-006	Haji Ali Junction	Haji Ali	GREEN	40	35	75	MODERATE	LOW	Haji Ali - Dr Annie Besant Road approach	MODERATE	MODERATE	20	Maintain Timing	Maintain current signal timing	40	35	0	0	Maintain stable intersection throughput	MONITORING	Traffic flow remains within normal operating range. Current timing is being monitored for changes in demand.	88.00	2026-09-08 19:47:32.419085	2026-09-08 19:47:32.419085
\.


--
-- TOC entry 5288 (class 0 OID 25420)
-- Dependencies: 221
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, username, password_hash, role, created_at) FROM stdin;
1	administrator	$2b$10$jZ6lAsWL9sa5X/UrS2.faeiKxbdMqXcCAuurCWc/rGOk2VzoFAnau	Administrator	2026-07-21 15:19:17.464467
2	mayor	$2b$10$jZ6lAsWL9sa5X/UrS2.faeiKxbdMqXcCAuurCWc/rGOk2VzoFAnau	Mayor	2026-07-21 15:19:17.467438
3	traffic_officer	$2b$10$jZ6lAsWL9sa5X/UrS2.faeiKxbdMqXcCAuurCWc/rGOk2VzoFAnau	Traffic Officer	2026-07-21 15:19:17.468092
4	police_officer	$2b$10$jZ6lAsWL9sa5X/UrS2.faeiKxbdMqXcCAuurCWc/rGOk2VzoFAnau	Police Officer	2026-07-21 15:19:17.468579
5	hospital_staff	$2b$10$jZ6lAsWL9sa5X/UrS2.faeiKxbdMqXcCAuurCWc/rGOk2VzoFAnau	Hospital Staff	2026-07-21 15:19:17.469184
6	fire_officer	$2b$10$jZ6lAsWL9sa5X/UrS2.faeiKxbdMqXcCAuurCWc/rGOk2VzoFAnau	Fire Officer	2026-07-21 15:19:17.469885
7	utility_officer	$2b$10$jZ6lAsWL9sa5X/UrS2.faeiKxbdMqXcCAuurCWc/rGOk2VzoFAnau	Utility Officer	2026-07-21 15:19:17.470335
8	citizen	$2b$10$jZ6lAsWL9sa5X/UrS2.faeiKxbdMqXcCAuurCWc/rGOk2VzoFAnau	Citizen	2026-07-21 15:19:17.470927
\.


--
-- TOC entry 5294 (class 0 OID 25619)
-- Dependencies: 227
-- Data for Name: vehicles; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vehicles (vehicle_id, vehicle_type, department, current_lat, current_lng, route_path, status, last_updated) FROM stdin;
\.


--
-- TOC entry 5376 (class 0 OID 0)
-- Dependencies: 252
-- Name: ambulance_operations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.ambulance_operations_id_seq', 2, true);


--
-- TOC entry 5377 (class 0 OID 0)
-- Dependencies: 248
-- Name: hospital_agent_decisions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.hospital_agent_decisions_id_seq', 1, false);


--
-- TOC entry 5378 (class 0 OID 0)
-- Dependencies: 246
-- Name: hospital_monitoring_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.hospital_monitoring_id_seq', 38, true);


--
-- TOC entry 5379 (class 0 OID 0)
-- Dependencies: 250
-- Name: hospital_resources_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.hospital_resources_id_seq', 17, true);


--
-- TOC entry 5380 (class 0 OID 0)
-- Dependencies: 231
-- Name: live_detection_results_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.live_detection_results_id_seq', 1, false);


--
-- TOC entry 5381 (class 0 OID 0)
-- Dependencies: 233
-- Name: live_event_timeline_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.live_event_timeline_id_seq', 1, true);


--
-- TOC entry 5382 (class 0 OID 0)
-- Dependencies: 237
-- Name: live_processing_logs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.live_processing_logs_id_seq', 1, false);


--
-- TOC entry 5383 (class 0 OID 0)
-- Dependencies: 235
-- Name: live_tracking_results_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.live_tracking_results_id_seq', 1, false);


--
-- TOC entry 5384 (class 0 OID 0)
-- Dependencies: 229
-- Name: live_video_uploads_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.live_video_uploads_id_seq', 2, true);


--
-- TOC entry 5385 (class 0 OID 0)
-- Dependencies: 222
-- Name: scenarios_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.scenarios_id_seq', 7, true);


--
-- TOC entry 5386 (class 0 OID 0)
-- Dependencies: 224
-- Name: timeline_events_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.timeline_events_id_seq', 30, true);


--
-- TOC entry 5387 (class 0 OID 0)
-- Dependencies: 239
-- Name: traffic_decisions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.traffic_decisions_id_seq', 65, true);


--
-- TOC entry 5388 (class 0 OID 0)
-- Dependencies: 241
-- Name: traffic_demo_scenarios_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.traffic_demo_scenarios_id_seq', 6, true);


--
-- TOC entry 5389 (class 0 OID 0)
-- Dependencies: 243
-- Name: traffic_signal_monitoring_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.traffic_signal_monitoring_id_seq', 199, true);


--
-- TOC entry 5390 (class 0 OID 0)
-- Dependencies: 220
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.users_id_seq', 8, true);


--
-- TOC entry 5124 (class 2606 OID 34277)
-- Name: ambulance_operations ambulance_operations_ambulance_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ambulance_operations
    ADD CONSTRAINT ambulance_operations_ambulance_id_key UNIQUE (ambulance_id);


--
-- TOC entry 5126 (class 2606 OID 34275)
-- Name: ambulance_operations ambulance_operations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ambulance_operations
    ADD CONSTRAINT ambulance_operations_pkey PRIMARY KEY (id);


--
-- TOC entry 5117 (class 2606 OID 34228)
-- Name: hospital_agent_decisions hospital_agent_decisions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_agent_decisions
    ADD CONSTRAINT hospital_agent_decisions_pkey PRIMARY KEY (id);


--
-- TOC entry 5113 (class 2606 OID 34209)
-- Name: hospital_monitoring hospital_monitoring_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_monitoring
    ADD CONSTRAINT hospital_monitoring_pkey PRIMARY KEY (id);


--
-- TOC entry 5120 (class 2606 OID 34256)
-- Name: hospital_resources hospital_resources_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_resources
    ADD CONSTRAINT hospital_resources_pkey PRIMARY KEY (id);


--
-- TOC entry 5110 (class 2606 OID 34177)
-- Name: hospitals hospitals_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospitals
    ADD CONSTRAINT hospitals_pkey PRIMARY KEY (hospital_id);


--
-- TOC entry 5080 (class 2606 OID 25649)
-- Name: incidents incidents_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.incidents
    ADD CONSTRAINT incidents_pkey PRIMARY KEY (incident_id);


--
-- TOC entry 5085 (class 2606 OID 25692)
-- Name: live_detection_results live_detection_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_detection_results
    ADD CONSTRAINT live_detection_results_pkey PRIMARY KEY (id);


--
-- TOC entry 5087 (class 2606 OID 25711)
-- Name: live_event_timeline live_event_timeline_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_event_timeline
    ADD CONSTRAINT live_event_timeline_pkey PRIMARY KEY (id);


--
-- TOC entry 5091 (class 2606 OID 25746)
-- Name: live_processing_logs live_processing_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_processing_logs
    ADD CONSTRAINT live_processing_logs_pkey PRIMARY KEY (id);


--
-- TOC entry 5089 (class 2606 OID 25728)
-- Name: live_tracking_results live_tracking_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_tracking_results
    ADD CONSTRAINT live_tracking_results_pkey PRIMARY KEY (id);


--
-- TOC entry 5083 (class 2606 OID 25678)
-- Name: live_video_uploads live_video_uploads_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_video_uploads
    ADD CONSTRAINT live_video_uploads_pkey PRIMARY KEY (id);


--
-- TOC entry 5075 (class 2606 OID 25618)
-- Name: map_entities map_entities_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.map_entities
    ADD CONSTRAINT map_entities_pkey PRIMARY KEY (entity_id);


--
-- TOC entry 5070 (class 2606 OID 25446)
-- Name: scenarios scenarios_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scenarios
    ADD CONSTRAINT scenarios_pkey PRIMARY KEY (id);


--
-- TOC entry 5073 (class 2606 OID 25460)
-- Name: timeline_events timeline_events_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.timeline_events
    ADD CONSTRAINT timeline_events_pkey PRIMARY KEY (id);


--
-- TOC entry 5095 (class 2606 OID 34037)
-- Name: traffic_decisions traffic_decisions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_decisions
    ADD CONSTRAINT traffic_decisions_pkey PRIMARY KEY (id);


--
-- TOC entry 5098 (class 2606 OID 34069)
-- Name: traffic_demo_scenarios traffic_demo_scenarios_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_demo_scenarios
    ADD CONSTRAINT traffic_demo_scenarios_name_key UNIQUE (name);


--
-- TOC entry 5100 (class 2606 OID 34067)
-- Name: traffic_demo_scenarios traffic_demo_scenarios_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_demo_scenarios
    ADD CONSTRAINT traffic_demo_scenarios_pkey PRIMARY KEY (id);


--
-- TOC entry 5106 (class 2606 OID 34112)
-- Name: traffic_signal_monitoring traffic_signal_monitoring_entity_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_signal_monitoring
    ADD CONSTRAINT traffic_signal_monitoring_entity_id_key UNIQUE (entity_id);


--
-- TOC entry 5108 (class 2606 OID 34110)
-- Name: traffic_signal_monitoring traffic_signal_monitoring_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_signal_monitoring
    ADD CONSTRAINT traffic_signal_monitoring_pkey PRIMARY KEY (id);


--
-- TOC entry 5066 (class 2606 OID 25431)
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- TOC entry 5068 (class 2606 OID 25433)
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- TOC entry 5078 (class 2606 OID 25634)
-- Name: vehicles vehicles_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vehicles
    ADD CONSTRAINT vehicles_pkey PRIMARY KEY (vehicle_id);


--
-- TOC entry 5127 (class 1259 OID 34283)
-- Name: idx_ambulance_hospital; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ambulance_hospital ON public.ambulance_operations USING btree (destination_hospital_id);


--
-- TOC entry 5128 (class 1259 OID 34284)
-- Name: idx_ambulance_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ambulance_status ON public.ambulance_operations USING btree (status);


--
-- TOC entry 5118 (class 1259 OID 34237)
-- Name: idx_hospital_decisions_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_hospital_decisions_type ON public.hospital_agent_decisions USING btree (decision_type);


--
-- TOC entry 5114 (class 1259 OID 34236)
-- Name: idx_hospital_monitoring_capacity; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_hospital_monitoring_capacity ON public.hospital_monitoring USING btree (capacity_level);


--
-- TOC entry 5115 (class 1259 OID 34235)
-- Name: idx_hospital_monitoring_hospital; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_hospital_monitoring_hospital ON public.hospital_monitoring USING btree (hospital_id);


--
-- TOC entry 5121 (class 1259 OID 34262)
-- Name: idx_hospital_resources_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_hospital_resources_status ON public.hospital_resources USING btree (status);


--
-- TOC entry 5122 (class 1259 OID 34263)
-- Name: idx_hospital_resources_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_hospital_resources_type ON public.hospital_resources USING btree (resource_type);


--
-- TOC entry 5111 (class 1259 OID 34234)
-- Name: idx_hospitals_area; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_hospitals_area ON public.hospitals USING btree (area);


--
-- TOC entry 5081 (class 1259 OID 33861)
-- Name: idx_live_video_uploads_video_hash; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_live_video_uploads_video_hash ON public.live_video_uploads USING btree (video_hash);


--
-- TOC entry 5101 (class 1259 OID 34118)
-- Name: idx_signal_monitoring_area; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_signal_monitoring_area ON public.traffic_signal_monitoring USING btree (area);


--
-- TOC entry 5102 (class 1259 OID 34121)
-- Name: idx_signal_monitoring_entity; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_signal_monitoring_entity ON public.traffic_signal_monitoring USING btree (entity_id);


--
-- TOC entry 5103 (class 1259 OID 34120)
-- Name: idx_signal_monitoring_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_signal_monitoring_status ON public.traffic_signal_monitoring USING btree (status);


--
-- TOC entry 5104 (class 1259 OID 34119)
-- Name: idx_signal_monitoring_traffic; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_signal_monitoring_traffic ON public.traffic_signal_monitoring USING btree (traffic_level);


--
-- TOC entry 5071 (class 1259 OID 25466)
-- Name: idx_timeline_events_timestamp; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_timeline_events_timestamp ON public.timeline_events USING btree (scenario_id, timestamp_second);


--
-- TOC entry 5092 (class 1259 OID 34038)
-- Name: idx_traffic_decisions_created_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_traffic_decisions_created_at ON public.traffic_decisions USING btree (created_at DESC);


--
-- TOC entry 5093 (class 1259 OID 34039)
-- Name: idx_traffic_decisions_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_traffic_decisions_type ON public.traffic_decisions USING btree (decision_type);


--
-- TOC entry 5096 (class 1259 OID 34070)
-- Name: idx_traffic_demo_scenarios_demo; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_traffic_demo_scenarios_demo ON public.traffic_demo_scenarios USING btree (is_demo, name);


--
-- TOC entry 5076 (class 1259 OID 34162)
-- Name: ux_map_entities_traffic_signal_code; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ux_map_entities_traffic_signal_code ON public.map_entities USING btree (((metadata ->> 'signal_code'::text))) WHERE (((entity_type)::text = 'traffic_signal'::text) AND ((metadata ->> 'signal_code'::text) IS NOT NULL));


--
-- TOC entry 5139 (class 2606 OID 34278)
-- Name: ambulance_operations ambulance_operations_destination_hospital_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ambulance_operations
    ADD CONSTRAINT ambulance_operations_destination_hospital_id_fkey FOREIGN KEY (destination_hospital_id) REFERENCES public.hospitals(hospital_id);


--
-- TOC entry 5137 (class 2606 OID 34229)
-- Name: hospital_agent_decisions hospital_agent_decisions_hospital_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_agent_decisions
    ADD CONSTRAINT hospital_agent_decisions_hospital_id_fkey FOREIGN KEY (hospital_id) REFERENCES public.hospitals(hospital_id) ON DELETE CASCADE;


--
-- TOC entry 5136 (class 2606 OID 34210)
-- Name: hospital_monitoring hospital_monitoring_hospital_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_monitoring
    ADD CONSTRAINT hospital_monitoring_hospital_id_fkey FOREIGN KEY (hospital_id) REFERENCES public.hospitals(hospital_id) ON DELETE CASCADE;


--
-- TOC entry 5138 (class 2606 OID 34257)
-- Name: hospital_resources hospital_resources_hospital_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_resources
    ADD CONSTRAINT hospital_resources_hospital_id_fkey FOREIGN KEY (hospital_id) REFERENCES public.hospitals(hospital_id) ON DELETE CASCADE;


--
-- TOC entry 5130 (class 2606 OID 33862)
-- Name: incidents incidents_upload_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.incidents
    ADD CONSTRAINT incidents_upload_id_fkey FOREIGN KEY (upload_id) REFERENCES public.live_video_uploads(id) ON DELETE SET NULL;


--
-- TOC entry 5131 (class 2606 OID 25693)
-- Name: live_detection_results live_detection_results_upload_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_detection_results
    ADD CONSTRAINT live_detection_results_upload_id_fkey FOREIGN KEY (upload_id) REFERENCES public.live_video_uploads(id) ON DELETE CASCADE;


--
-- TOC entry 5132 (class 2606 OID 25712)
-- Name: live_event_timeline live_event_timeline_upload_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_event_timeline
    ADD CONSTRAINT live_event_timeline_upload_id_fkey FOREIGN KEY (upload_id) REFERENCES public.live_video_uploads(id) ON DELETE CASCADE;


--
-- TOC entry 5134 (class 2606 OID 25747)
-- Name: live_processing_logs live_processing_logs_upload_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_processing_logs
    ADD CONSTRAINT live_processing_logs_upload_id_fkey FOREIGN KEY (upload_id) REFERENCES public.live_video_uploads(id) ON DELETE CASCADE;


--
-- TOC entry 5133 (class 2606 OID 25729)
-- Name: live_tracking_results live_tracking_results_upload_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.live_tracking_results
    ADD CONSTRAINT live_tracking_results_upload_id_fkey FOREIGN KEY (upload_id) REFERENCES public.live_video_uploads(id) ON DELETE CASCADE;


--
-- TOC entry 5129 (class 2606 OID 25461)
-- Name: timeline_events timeline_events_scenario_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.timeline_events
    ADD CONSTRAINT timeline_events_scenario_id_fkey FOREIGN KEY (scenario_id) REFERENCES public.scenarios(id) ON DELETE CASCADE;


--
-- TOC entry 5135 (class 2606 OID 34113)
-- Name: traffic_signal_monitoring traffic_signal_monitoring_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_signal_monitoring
    ADD CONSTRAINT traffic_signal_monitoring_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES public.map_entities(entity_id) ON DELETE CASCADE;


--
-- TOC entry 5327 (class 0 OID 0)
-- Dependencies: 253
-- Name: TABLE ambulance_operations; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.ambulance_operations TO smart_city_user;


--
-- TOC entry 5329 (class 0 OID 0)
-- Dependencies: 252
-- Name: SEQUENCE ambulance_operations_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.ambulance_operations_id_seq TO smart_city_user;


--
-- TOC entry 5330 (class 0 OID 0)
-- Dependencies: 249
-- Name: TABLE hospital_agent_decisions; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.hospital_agent_decisions TO smart_city_user;


--
-- TOC entry 5332 (class 0 OID 0)
-- Dependencies: 248
-- Name: SEQUENCE hospital_agent_decisions_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.hospital_agent_decisions_id_seq TO smart_city_user;


--
-- TOC entry 5333 (class 0 OID 0)
-- Dependencies: 247
-- Name: TABLE hospital_monitoring; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.hospital_monitoring TO smart_city_user;


--
-- TOC entry 5335 (class 0 OID 0)
-- Dependencies: 246
-- Name: SEQUENCE hospital_monitoring_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.hospital_monitoring_id_seq TO smart_city_user;


--
-- TOC entry 5336 (class 0 OID 0)
-- Dependencies: 251
-- Name: TABLE hospital_resources; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.hospital_resources TO smart_city_user;


--
-- TOC entry 5338 (class 0 OID 0)
-- Dependencies: 250
-- Name: SEQUENCE hospital_resources_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.hospital_resources_id_seq TO smart_city_user;


--
-- TOC entry 5339 (class 0 OID 0)
-- Dependencies: 245
-- Name: TABLE hospitals; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.hospitals TO smart_city_user;


--
-- TOC entry 5340 (class 0 OID 0)
-- Dependencies: 228
-- Name: TABLE incidents; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.incidents TO smart_city_user;


--
-- TOC entry 5341 (class 0 OID 0)
-- Dependencies: 232
-- Name: TABLE live_detection_results; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.live_detection_results TO smart_city_user;


--
-- TOC entry 5343 (class 0 OID 0)
-- Dependencies: 231
-- Name: SEQUENCE live_detection_results_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.live_detection_results_id_seq TO smart_city_user;


--
-- TOC entry 5344 (class 0 OID 0)
-- Dependencies: 234
-- Name: TABLE live_event_timeline; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.live_event_timeline TO smart_city_user;


--
-- TOC entry 5346 (class 0 OID 0)
-- Dependencies: 233
-- Name: SEQUENCE live_event_timeline_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.live_event_timeline_id_seq TO smart_city_user;


--
-- TOC entry 5347 (class 0 OID 0)
-- Dependencies: 238
-- Name: TABLE live_processing_logs; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.live_processing_logs TO smart_city_user;


--
-- TOC entry 5349 (class 0 OID 0)
-- Dependencies: 237
-- Name: SEQUENCE live_processing_logs_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.live_processing_logs_id_seq TO smart_city_user;


--
-- TOC entry 5350 (class 0 OID 0)
-- Dependencies: 236
-- Name: TABLE live_tracking_results; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.live_tracking_results TO smart_city_user;


--
-- TOC entry 5352 (class 0 OID 0)
-- Dependencies: 235
-- Name: SEQUENCE live_tracking_results_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.live_tracking_results_id_seq TO smart_city_user;


--
-- TOC entry 5353 (class 0 OID 0)
-- Dependencies: 230
-- Name: TABLE live_video_uploads; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.live_video_uploads TO smart_city_user;


--
-- TOC entry 5355 (class 0 OID 0)
-- Dependencies: 229
-- Name: SEQUENCE live_video_uploads_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.live_video_uploads_id_seq TO smart_city_user;


--
-- TOC entry 5356 (class 0 OID 0)
-- Dependencies: 226
-- Name: TABLE map_entities; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.map_entities TO smart_city_user;


--
-- TOC entry 5357 (class 0 OID 0)
-- Dependencies: 223
-- Name: TABLE scenarios; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.scenarios TO smart_city_user;


--
-- TOC entry 5359 (class 0 OID 0)
-- Dependencies: 222
-- Name: SEQUENCE scenarios_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.scenarios_id_seq TO smart_city_user;


--
-- TOC entry 5360 (class 0 OID 0)
-- Dependencies: 225
-- Name: TABLE timeline_events; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.timeline_events TO smart_city_user;


--
-- TOC entry 5362 (class 0 OID 0)
-- Dependencies: 224
-- Name: SEQUENCE timeline_events_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.timeline_events_id_seq TO smart_city_user;


--
-- TOC entry 5363 (class 0 OID 0)
-- Dependencies: 240
-- Name: TABLE traffic_decisions; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.traffic_decisions TO smart_city_user;


--
-- TOC entry 5365 (class 0 OID 0)
-- Dependencies: 239
-- Name: SEQUENCE traffic_decisions_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.traffic_decisions_id_seq TO smart_city_user;


--
-- TOC entry 5366 (class 0 OID 0)
-- Dependencies: 242
-- Name: TABLE traffic_demo_scenarios; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.traffic_demo_scenarios TO smart_city_user;


--
-- TOC entry 5368 (class 0 OID 0)
-- Dependencies: 241
-- Name: SEQUENCE traffic_demo_scenarios_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.traffic_demo_scenarios_id_seq TO smart_city_user;


--
-- TOC entry 5369 (class 0 OID 0)
-- Dependencies: 244
-- Name: TABLE traffic_signal_monitoring; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.traffic_signal_monitoring TO smart_city_user;


--
-- TOC entry 5371 (class 0 OID 0)
-- Dependencies: 243
-- Name: SEQUENCE traffic_signal_monitoring_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.traffic_signal_monitoring_id_seq TO smart_city_user;


--
-- TOC entry 5372 (class 0 OID 0)
-- Dependencies: 221
-- Name: TABLE users; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.users TO smart_city_user;


--
-- TOC entry 5374 (class 0 OID 0)
-- Dependencies: 220
-- Name: SEQUENCE users_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.users_id_seq TO smart_city_user;


--
-- TOC entry 5375 (class 0 OID 0)
-- Dependencies: 227
-- Name: TABLE vehicles; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.vehicles TO smart_city_user;


--
-- TOC entry 2177 (class 826 OID 25384)
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: public; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON SEQUENCES TO smart_city_user;


--
-- TOC entry 2176 (class 826 OID 25383)
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: public; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO smart_city_user;


-- Completed on 2026-10-02 15:55:23

--
-- PostgreSQL database dump complete
--

\unrestrict 8yqyVAJdsgjbf43hQDfmMs3yWGMu09e1LpCTXeeLgnga06PevIEpBtMaEq3hEdC

