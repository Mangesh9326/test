--
-- PostgreSQL database dump
--

\restrict dXT3aoqakj9BHSpL2Ua6GGnuQ1zG6Wh78CvInuapy16zoB3NbwpCImJWG4KciEX

-- Dumped from database version 18.4
-- Dumped by pg_dump version 18.4

-- Started on 2026-09-08 19:45:31

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
-- TOC entry 5123 (class 0 OID 0)
-- Dependencies: 243
-- Name: traffic_signal_monitoring_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.traffic_signal_monitoring_id_seq OWNED BY public.traffic_signal_monitoring.id;


--
-- TOC entry 4944 (class 2604 OID 34081)
-- Name: traffic_signal_monitoring id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_signal_monitoring ALTER COLUMN id SET DEFAULT nextval('public.traffic_signal_monitoring_id_seq'::regclass);


--
-- TOC entry 5116 (class 0 OID 34078)
-- Dependencies: 244
-- Data for Name: traffic_signal_monitoring; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.traffic_signal_monitoring (id, entity_id, signal_code, intersection_name, area, current_phase, current_green_seconds, current_red_seconds, cycle_time_seconds, traffic_level, density_level, affected_approach, predicted_traffic_level, predicted_density_level, prediction_horizon_minutes, ai_decision, ai_recommendation, recommended_green_seconds, recommended_red_seconds, green_adjustment_seconds, red_adjustment_seconds, expected_impact, status, reason, confidence, last_observed_at, updated_at) FROM stdin;
1	af67a23b-986a-4372-a405-eaecf1f88565	SIG-af67a2	Dadar Junction	Dadar	GREEN	58	34	92	MODERATE	MODERATE	\N	\N	\N	15	Monitoring	No change	\N	\N	0	0	\N	MONITORING	\N	\N	2026-09-08 16:53:56.396217	2026-09-08 17:07:41.526238
2	978f1968-9211-4173-8c3f-02e5a7ef9742	SIG-978f19	Wadala Junction	Wadala	GREEN	33	26	59	MODERATE	LOW	\N	\N	\N	15	Monitoring	No change	\N	\N	0	0	\N	MONITORING	\N	\N	2026-09-08 16:53:56.396217	2026-09-08 17:07:41.526238
3	f051e67e-a2ce-4d43-a2e2-d48bb87e01b3	SIG-f051e6	Sion Circle	Sion	GREEN	47	44	91	SEVERE	LOW	\N	\N	\N	15	Monitoring	No change	\N	\N	0	0	\N	MONITORING	\N	\N	2026-09-08 16:53:56.396217	2026-09-08 17:07:41.526238
4	786fd97f-5099-41b5-b5c9-d1fc89d1ed4e	SIG-786fd9	Matunga Junction	Matunga	GREEN	36	28	64	SEVERE	LOW	\N	\N	\N	15	Monitoring	No change	\N	\N	0	0	\N	MONITORING	\N	\N	2026-09-08 16:53:56.396217	2026-09-08 17:07:41.526238
5	17a7ee5c-ab37-4fe4-b99e-013c4cea85d2	SIG-17a7ee	Bandra Junction	Bandra	GREEN	59	44	103	HIGH	LOW	\N	\N	\N	15	Monitoring	No change	\N	\N	0	0	\N	MONITORING	\N	\N	2026-09-08 16:53:56.396217	2026-09-08 17:07:41.526238
6	9ea42355-89ca-46f0-b58b-6a37714d64fd	SIG-9ea423	BKC Junction	BKC	GREEN	38	44	82	MODERATE	HIGH	\N	\N	\N	15	Monitoring	No change	\N	\N	0	0	\N	MONITORING	\N	\N	2026-09-08 16:53:56.396217	2026-09-08 17:07:41.526238
7	894ff76c-a263-4ed0-9c00-5a3060440eba	SIG-894ff7	Worli Junction	Worli	GREEN	37	42	79	MODERATE	MODERATE	\N	\N	\N	15	Monitoring	No change	\N	\N	0	0	\N	MONITORING	\N	\N	2026-09-08 16:53:56.396217	2026-09-08 17:07:41.526238
8	6195470e-4031-4f89-8857-5ce348d50f04	SIG-SOU-001	Hutatma Chowk Junction	Fort	GREEN	40	35	75	HIGH	LOW	Primary inbound approach	SEVERE	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Reduce queue accumulation	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	85.78	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
9	5384a3cd-3839-42b6-bbf4-fb8d3c2fc0f7	SIG-SOU-002	CSMT Junction	Fort	RED	45	30	75	MODERATE	LOW	Primary inbound approach	SEVERE	MODERATE	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	85.06	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
10	622338b9-b794-46c0-9b95-1ece33df3d8c	SIG-SOU-003	Churchgate Junction	Churchgate	RED	45	30	75	MODERATE	LOW	Primary inbound approach	MODERATE	LOW	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	95.75	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
11	5883c94a-fa0f-4a78-9ee0-c5f9c4b273b7	SIG-SOU-004	Marine Drive Junction	Marine Drive	GREEN	35	25	60	SEVERE	LOW	Primary inbound approach	SEVERE	MODERATE	15	Increase Green	No change	\N	\N	0	0	Maintain intersection throughput	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	92.69	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
12	4407747f-e360-42c0-bfa6-df2afdc27c2a	SIG-SOU-005	Nariman Point Junction	Nariman Point	RED	40	30	70	HIGH	MODERATE	Cross traffic approach	HIGH	LOW	15	Maintain Timing	Coordinate next cycle	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	94.83	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
13	ec145b30-c966-49a4-ab9b-3285f6695c02	SIG-SOU-006	Haji Ali Junction	Haji Ali	RED	40	30	70	MODERATE	LOW	Cross traffic approach	MODERATE	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	91.45	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
14	fde330ce-138f-4072-ae10-051435acfedc	SIG-SOU-007	Mahalaxmi Junction	Mahalaxmi	GREEN	35	25	60	LOW	MODERATE	Primary inbound approach	HIGH	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	90.80	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
15	536ccf71-174d-451d-ad0c-d6143c692292	SIG-SOU-008	Tardeo Junction	Tardeo	RED	35	30	65	LOW	MODERATE	Cross traffic approach	CRITICAL	MODERATE	15	Increase Green	Coordinate next cycle	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	91.52	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
16	0773e544-c4b3-405f-afdc-e76f70f8af09	SIG-SOU-009	Mumbai Central Junction	Mumbai Central	RED	40	35	75	HIGH	LOW	Cross traffic approach	MODERATE	MODERATE	15	Maintain Timing	Green +5s / Red -5s	\N	\N	0	0	Reduce queue accumulation	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	91.36	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
17	0eb75675-e91f-4484-90b4-f0bc2cbb6cd7	SIG-SOU-010	Byculla Junction	Byculla	GREEN	35	25	60	MODERATE	LOW	Cross traffic approach	CRITICAL	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	88.91	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
18	bc276040-2430-41fb-b1ba-2cecf867a4e2	SIG-WOR-001	Worli Naka Junction	Worli	RED	45	35	80	MODERATE	LOW	Primary inbound approach	SEVERE	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	85.93	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
19	19ddd422-42ce-4da2-bace-7a3335b3b4cc	SIG-WOR-002	Worli Sea Face Junction	Worli	RED	45	35	80	MODERATE	LOW	Cross traffic approach	MODERATE	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	87.90	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
20	96e5b191-b4f4-4dd1-9843-ae18b746fc19	SIG-WOR-003	Prabhadevi Junction	Prabhadevi	GREEN	45	40	85	MODERATE	LOW	Cross traffic approach	SEVERE	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	93.28	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
21	8e9bff0f-f9d6-4318-9030-555a7c097969	SIG-WOR-004	Lower Parel Junction	Lower Parel	RED	45	35	80	SEVERE	LOW	Primary inbound approach	HIGH	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	87.11	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
22	c6cfb255-4dd2-4223-a1c3-79170c77ecf5	SIG-WOR-005	Elphinstone Junction	Elphinstone	RED	45	35	80	MODERATE	LOW	Primary inbound approach	MODERATE	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	94.36	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
23	33638101-04ca-4bed-839f-7702216108ba	SIG-DAD-001	Dadar TT Circle	Dadar	GREEN	40	25	65	MODERATE	MODERATE	Cross traffic approach	MODERATE	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	93.01	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
24	a7d603ae-72c4-4905-b0db-e012d89a2e46	SIG-DAD-002	Dadar Shivaji Park Junction	Dadar	RED	40	35	75	HIGH	LOW	Cross traffic approach	MODERATE	LOW	15	Coordinate Corridor	Green +5s / Red -5s	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	92.89	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
25	c5980fde-e36e-4c20-b660-febb435cbc54	SIG-MAH-001	Mahim Junction	Mahim	RED	40	35	75	MODERATE	LOW	Cross traffic approach	CRITICAL	LOW	15	Maintain Timing	No change	\N	\N	0	0	Reduce queue accumulation	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	93.09	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
26	37c451e1-2bcf-4202-8d16-7b33fe9b8abc	SIG-MAH-002	Mahim Causeway Junction	Mahim	GREEN	35	35	70	HIGH	HIGH	Cross traffic approach	HIGH	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	94.53	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
27	d803906e-7f3e-4276-847a-2c1137d7b38c	SIG-MAT-001	Matunga Five Gardens Junction	Matunga	RED	35	30	65	LOW	MODERATE	Primary inbound approach	HIGH	LOW	15	Increase Green	No change	\N	\N	0	0	Reduce queue accumulation	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	85.42	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
28	39106148-7a6b-4be2-8941-a1c0f8b38338	SIG-SIO-001	Sion Circle	Sion	RED	40	30	70	MODERATE	LOW	Primary inbound approach	SEVERE	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	91.65	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
29	cddef124-03d6-4080-ba9e-5d0d13186e04	SIG-SIO-002	Sion Hospital Junction	Sion	GREEN	45	35	80	MODERATE	MODERATE	Primary inbound approach	MODERATE	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	91.16	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
30	cd8353e5-35f3-4908-8bd8-7469bea036f7	SIG-BAN-001	Bandra Junction	Bandra	RED	35	25	60	MODERATE	LOW	Cross traffic approach	CRITICAL	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	88.53	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
31	d917ad96-6f35-4d74-88f3-145bce83bcd1	SIG-BAN-002	Bandra Bandstand Junction	Bandra	RED	40	40	80	HIGH	MODERATE	Primary inbound approach	HIGH	LOW	15	Maintain Timing	No change	\N	\N	0	0	Reduce queue accumulation	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	87.08	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
32	0f254671-b2d1-4595-833d-f5c6ccbdca98	SIG-BAN-003	Khar Junction	Khar	GREEN	45	40	85	HIGH	LOW	Cross traffic approach	SEVERE	HIGH	15	Coordinate Corridor	Green +5s / Red -5s	\N	\N	0	0	Maintain intersection throughput	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	88.62	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
33	ad66ef0b-eb35-438a-a350-e39aef911b67	SIG-SAN-001	Santacruz Junction	Santacruz	RED	35	30	65	HIGH	LOW	Primary inbound approach	HIGH	MODERATE	15	Increase Green	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	90.39	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
34	b9bd40cb-44cc-45f7-a00d-2197ce85ec0b	SIG-SAN-002	Vakola Junction	Santacruz	RED	35	30	65	MODERATE	HIGH	Cross traffic approach	HIGH	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	88.98	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
35	bb879fcb-3700-41a4-a25a-6fcfe4cce058	SIG-SAN-003	Milan Subway Junction	Santacruz	GREEN	45	25	70	SEVERE	LOW	Cross traffic approach	MODERATE	LOW	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	92.84	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
36	b50d7246-f7f4-4e73-a13a-373c4faf2a3b	SIG-BKC-001	BKC Kalanagar Junction	BKC	RED	35	40	75	LOW	LOW	Cross traffic approach	HIGH	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	90.20	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
37	b40fdc72-96e2-4048-9a69-5086b61fb44a	SIG-BKC-002	BKC Bharat Diamond Bourse Junction	BKC	RED	35	30	65	LOW	LOW	Primary inbound approach	SEVERE	LOW	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	94.16	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
38	e49660f6-23f0-4f98-998e-5692e62dba62	SIG-BKC-003	BKC Kurla Junction	BKC	GREEN	45	30	75	LOW	MODERATE	Cross traffic approach	CRITICAL	LOW	15	Coordinate Corridor	Coordinate next cycle	\N	\N	0	0	Maintain intersection throughput	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	91.05	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
39	28ec1ca1-fc65-45fa-ad45-018c15001636	SIG-AND-001	Andheri Station Junction	Andheri	RED	40	35	75	SEVERE	LOW	Primary inbound approach	SEVERE	LOW	15	Maintain Timing	Coordinate next cycle	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	87.34	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
40	b2d15ebb-886b-497f-950a-17cc46ae430a	SIG-AND-002	Andheri Market Junction	Andheri	RED	40	30	70	SEVERE	LOW	Cross traffic approach	MODERATE	LOW	15	Maintain Timing	No change	\N	\N	0	0	Reduce queue accumulation	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	85.59	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
41	5691f2d5-fbe4-4c22-a9c1-3d5ef0a7ebb8	SIG-AND-003	Andheri Kurla Junction	Andheri	GREEN	40	40	80	MODERATE	LOW	Cross traffic approach	MODERATE	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	92.05	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
42	99513606-d20e-4938-9646-417b96a52a80	SIG-AND-004	Chakala Junction	Andheri	RED	45	30	75	LOW	MODERATE	Cross traffic approach	HIGH	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	89.26	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
43	117791e3-f16d-4c09-bdec-ad2514c31cde	SIG-AND-005	Marol Naka Junction	Marol	RED	40	30	70	HIGH	LOW	Primary inbound approach	MODERATE	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	94.99	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
44	cb673f52-23d2-43e5-8333-383c518c8ee7	SIG-AND-006	Saki Naka Junction	Saki Naka	GREEN	50	35	85	HIGH	LOW	Primary inbound approach	SEVERE	LOW	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	94.92	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
45	9bfcaf41-1f05-47a7-a55b-c4206b504378	SIG-JOG-001	Jogeshwari Junction	Jogeshwari	RED	40	25	65	SEVERE	MODERATE	Cross traffic approach	SEVERE	LOW	15	Maintain Timing	Green +5s / Red -5s	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	87.84	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
46	85167a8d-e080-4bb1-ad94-d84e00bc8912	SIG-JOG-002	JVLR Western Junction	Jogeshwari	RED	40	35	75	MODERATE	LOW	Cross traffic approach	MODERATE	LOW	15	Maintain Timing	Green +5s / Red -5s	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	91.51	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
47	80b13ccf-8b0a-49a4-9903-9a5f9739037c	SIG-GOR-001	Goregaon Junction	Goregaon	GREEN	40	30	70	MODERATE	MODERATE	Cross traffic approach	HIGH	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	89.66	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
48	19e54f28-8d6c-4a98-96c0-fc14b2e62590	SIG-GOR-002	Goregaon WEH Junction	Goregaon	RED	45	40	85	MODERATE	LOW	Primary inbound approach	HIGH	MODERATE	15	Maintain Timing	Green +5s / Red -5s	\N	\N	0	0	Reduce queue accumulation	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	92.01	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
49	8fc58d1f-e4ef-4e11-8326-5ccae1104aee	SIG-GOR-003	Aarey Junction	Aarey	RED	35	25	60	CRITICAL	LOW	Primary inbound approach	MODERATE	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	89.38	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
50	1c7d00d0-c315-4bdb-be93-545d408ef92f	SIG-MAL-001	Malad Junction	Malad	GREEN	40	25	65	LOW	HIGH	Cross traffic approach	HIGH	LOW	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	93.45	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
51	b4fa4333-7ccd-4973-8390-fb9c073f7538	SIG-MAL-002	Malad Link Road Junction	Malad	RED	40	30	70	MODERATE	MODERATE	Cross traffic approach	HIGH	MODERATE	15	Maintain Timing	Coordinate next cycle	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	96.56	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
52	7eaa9651-3978-4d0c-93db-af1aa900006b	SIG-KAN-001	Kandivali Junction	Kandivali	RED	35	25	60	MODERATE	MODERATE	Cross traffic approach	HIGH	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Reduce queue accumulation	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	95.75	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
53	a295064f-9b20-4116-a083-a0a6db2af40f	SIG-BOR-001	Borivali Junction	Borivali	GREEN	40	35	75	SEVERE	LOW	Cross traffic approach	MODERATE	LOW	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	94.24	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
54	20176dbe-c336-452d-b582-c33bc641e8ea	SIG-BOR-002	Borivali WEH Junction	Borivali	RED	45	30	75	MODERATE	LOW	Cross traffic approach	HIGH	LOW	15	Increase Green	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	96.76	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
55	550c3b5d-c034-4f6f-8c11-ad2012618545	SIG-KUR-001	Kurla Depot Junction	Kurla	RED	40	25	65	LOW	MODERATE	Primary inbound approach	CRITICAL	LOW	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	90.99	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
56	67ca2e85-012b-40da-a559-24dd4f9ebc92	SIG-KUR-002	Kurla Station Junction	Kurla	GREEN	40	40	80	MODERATE	LOW	Primary inbound approach	HIGH	MODERATE	15	Maintain Timing	Green +5s / Red -5s	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	95.52	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
57	7adb3499-e4da-425a-b1be-52de6d8ceeeb	SIG-GHA-001	Ghatkopar Junction	Ghatkopar	RED	40	30	70	HIGH	MODERATE	Cross traffic approach	HIGH	HIGH	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	95.48	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
58	33751247-ce67-46e8-89dd-4dfd9199b295	SIG-GHA-002	Ghatkopar West Junction	Ghatkopar	RED	45	35	80	MODERATE	MODERATE	Primary inbound approach	MODERATE	LOW	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	90.45	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
59	d371eb4a-fc63-4dd7-a75a-fd2012a1f201	SIG-VIK-001	Vikhroli Junction	Vikhroli	GREEN	40	35	75	HIGH	HIGH	Cross traffic approach	HIGH	LOW	15	Increase Green	Coordinate next cycle	\N	\N	0	0	Reduce queue accumulation	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	85.01	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
60	6fefba03-216d-48d1-bba6-1c7f89de2615	SIG-VIK-002	Vikhroli Parksite Junction	Vikhroli	RED	40	30	70	HIGH	LOW	Primary inbound approach	MODERATE	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	96.41	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
61	8a0f30a0-9057-47d3-8c05-07a252bfd48e	SIG-BHA-001	Bhandup Junction	Bhandup	RED	40	25	65	LOW	LOW	Cross traffic approach	HIGH	MODERATE	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	89.10	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
62	7ed2754f-d89d-434c-94ff-fb109924b9c7	SIG-MUL-001	Mulund Check Naka	Mulund	GREEN	35	35	70	SEVERE	LOW	Primary inbound approach	MODERATE	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	90.68	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
63	a75b7c65-7524-49bd-9a37-3db44901a55d	SIG-CHE-001	Chembur Junction	Chembur	RED	45	30	75	MODERATE	LOW	Cross traffic approach	MODERATE	MODERATE	15	Maintain Timing	Coordinate next cycle	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	86.89	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
64	65aa004f-a2b2-4eac-84a9-c09fb619a789	SIG-CHE-002	Amar Mahal Junction	Chembur	RED	40	25	65	MODERATE	LOW	Primary inbound approach	MODERATE	HIGH	15	Increase Green	No change	\N	\N	0	0	Reduce queue accumulation	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	89.80	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
65	ce390c23-f218-4ea4-a68b-2a9caa84a093	SIG-CHE-003	RCF Junction	Chembur	GREEN	40	30	70	MODERATE	LOW	Primary inbound approach	MODERATE	LOW	15	Coordinate Corridor	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	88.13	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
66	f29ef3ee-b345-4bf8-9e69-039bb660b725	SIG-CHE-004	Vashi Naka Junction	Chembur	RED	35	30	65	LOW	LOW	Cross traffic approach	HIGH	MODERATE	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	93.07	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
67	f20d118f-085d-4c93-a693-d2fa999ad809	SIG-WAD-001	Wadala Junction	Wadala	RED	40	35	75	HIGH	LOW	Cross traffic approach	SEVERE	LOW	15	Coordinate Corridor	Coordinate next cycle	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	85.07	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
68	ac0f6750-b632-459c-ad10-f9c89e684605	SIG-WAD-002	Wadala IMAX Junction	Wadala	GREEN	50	35	85	MODERATE	MODERATE	Primary inbound approach	CRITICAL	LOW	15	Maintain Timing	No change	\N	\N	0	0	Maintain intersection throughput	RECOMMENDATION READY	Traffic conditions evaluated from current signal monitoring inputs.	86.06	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
69	d5345fab-0739-4f93-9193-58afae8af705	SIG-WAD-003	Antop Hill Junction	Antop Hill	RED	50	35	85	LOW	MODERATE	Primary inbound approach	MODERATE	LOW	15	Increase Green	Green +5s / Red -5s	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	86.52	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
70	d2e6c6f9-e4c1-4902-968a-9f27699b3748	SIG-MAN-001	Mankhurd Junction	Mankhurd	RED	35	30	65	HIGH	LOW	Cross traffic approach	MODERATE	LOW	15	Increase Green	No change	\N	\N	0	0	Maintain intersection throughput	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	86.67	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
71	8a580f56-7a2a-4c15-a552-dce5a0e4f7d1	SIG-GOV-001	Govandi Junction	Govandi	GREEN	40	25	65	HIGH	LOW	Primary inbound approach	MODERATE	LOW	15	Maintain Timing	Coordinate next cycle	\N	\N	0	0	Reduce queue accumulation	MONITORING	Traffic conditions evaluated from current signal monitoring inputs.	95.09	2026-09-08 17:07:31.865309	2026-09-08 17:07:41.526238
\.


--
-- TOC entry 5125 (class 0 OID 0)
-- Dependencies: 243
-- Name: traffic_signal_monitoring_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.traffic_signal_monitoring_id_seq', 199, true);


--
-- TOC entry 4964 (class 2606 OID 34112)
-- Name: traffic_signal_monitoring traffic_signal_monitoring_entity_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_signal_monitoring
    ADD CONSTRAINT traffic_signal_monitoring_entity_id_key UNIQUE (entity_id);


--
-- TOC entry 4966 (class 2606 OID 34110)
-- Name: traffic_signal_monitoring traffic_signal_monitoring_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_signal_monitoring
    ADD CONSTRAINT traffic_signal_monitoring_pkey PRIMARY KEY (id);


--
-- TOC entry 4959 (class 1259 OID 34118)
-- Name: idx_signal_monitoring_area; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_signal_monitoring_area ON public.traffic_signal_monitoring USING btree (area);


--
-- TOC entry 4960 (class 1259 OID 34121)
-- Name: idx_signal_monitoring_entity; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_signal_monitoring_entity ON public.traffic_signal_monitoring USING btree (entity_id);


--
-- TOC entry 4961 (class 1259 OID 34120)
-- Name: idx_signal_monitoring_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_signal_monitoring_status ON public.traffic_signal_monitoring USING btree (status);


--
-- TOC entry 4962 (class 1259 OID 34119)
-- Name: idx_signal_monitoring_traffic; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_signal_monitoring_traffic ON public.traffic_signal_monitoring USING btree (traffic_level);


--
-- TOC entry 4967 (class 2606 OID 34113)
-- Name: traffic_signal_monitoring traffic_signal_monitoring_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.traffic_signal_monitoring
    ADD CONSTRAINT traffic_signal_monitoring_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES public.map_entities(entity_id) ON DELETE CASCADE;


--
-- TOC entry 5122 (class 0 OID 0)
-- Dependencies: 244
-- Name: TABLE traffic_signal_monitoring; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.traffic_signal_monitoring TO smart_city_user;


--
-- TOC entry 5124 (class 0 OID 0)
-- Dependencies: 243
-- Name: SEQUENCE traffic_signal_monitoring_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.traffic_signal_monitoring_id_seq TO smart_city_user;


-- Completed on 2026-09-08 19:45:31

--
-- PostgreSQL database dump complete
--

\unrestrict dXT3aoqakj9BHSpL2Ua6GGnuQ1zG6Wh78CvInuapy16zoB3NbwpCImJWG4KciEX

