import React, { useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import MainLayout from "./layouts/MainLayout";
import DepartmentLayout from "./layouts/DepartmentLayout";
import socketService from "./services/socketService";
import { departmentSidebars } from "./navigation/navigationConfig";

// Import Pages
import DemonstrationInput from "./pages/DemonstrationInput";
import UnifiedDashboard from "./pages/Dashboard/UnifiedDashboard";
import DigitalTwinMap from "./components/DigitalTwin/MapWidget";
import SimulateDemonstation from "./pages/SimulateDemonstation";
//Traffic Pages
import TrafficOverview from "./pages/Departments/Traffic/Overview";
import LiveTraffic from "./pages/Departments/Traffic/LiveTraffic";
import RouteIntelligence from "./pages/Departments/Traffic/RouteIntelligence";
import TrafficSignals from "./pages/Departments/Traffic/TrafficSignals";
import TrafficCameras from "./pages/Departments/Traffic/TrafficCameras2";
import EmergencyPriority from "./pages/Departments/Traffic/EmergencyPriority";
import CongestionHeatmap from "./pages/Departments/Traffic/CongestionHeatmap";
import AIPredictions from "./pages/Departments/Traffic/AIPredictions";
//Hospital Pages
import HospitalOverview from "./pages/Departments/Hospital/Overview";
import Hospitals from "./pages/Departments/Hospital/Hospitals";
import BedAvailability from "./pages/Departments/Hospital/BedAvailability";
import HospitalCapacity from "./pages/Departments/Hospital/HospitalCapacity";
import EmergencyDepartment from "./pages/Departments/Hospital/EmergencyDepartment";
import PatientFlow from "./pages/Departments/Hospital/PatientFlow";
import ResourceManagement from "./pages/Departments/Hospital/ResourceManagement";
import EmergencyAmbulance from "./pages/Departments/Hospital/EmergencyAmbulance";
import HospitalAgent from "./pages/Departments/Hospital/HospitalAgent";
// Police Pages
import PoliceOverview from "./pages/Departments/Police/Overview";
import PoliceIncidents from "./pages/Departments/Police/incidents/Incidents";
import CrimeMonitoring from "./pages/Departments/Police/CrimeMonitoring";
import PatrolOperations from "./pages/Departments/Police/PatrolOperations";
import PoliceUnits from "./pages/Departments/Police/PoliceUnits";
import CCTVIntelligence from "./pages/Departments/Police/CCTVIntelligence";
import PoliceEmergencyResponse from "./pages/Departments/Police/EmergencyResponse";
import PoliceAgent from "./pages/Departments/Police/Agent";
//Fire pages
import FireOverview from "./pages/Departments/Fire/Overview";
import FireIncidents from "./pages/Departments/Fire/Incidents";
import FireResponseDispatch from "./pages/Departments/Fire/ResponseDispatch";
import FireStations from "./pages/Departments/Fire/FireStations";
import FireHydrants from "./pages/Departments/Fire/HydrantWaterSupply";
import FireAgent from "./pages/Departments/Fire/Agent";
import FireAnalytics from "./pages/Departments/Fire/FireAnalytics";
// Citizen pages
import CitizenLayout from "./pages/Citizen/CitizenLayout";
import CitizenAroundMe from "./pages/Citizen/AroundMe";
import CitizenTraffic from "./pages/Citizen/Traffic";
import CitizenAlerts from "./pages/Citizen/Alerts";
import CitizenServices from "./pages/Citizen/Services";
import CitizenAssistant from "./pages/Citizen/Assistant";

// Placeholder Pages for scaffolding
const PlaceholderPage = ({ title }) => (
  <div className="flex h-full items-center justify-center text-city-700 text-xl font-bold">
    [{title} Module Offline]
  </div>
);

export default function App() {
  useEffect(() => {
    socketService.connect();
    return () => socketService.disconnect();
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="simulate-scenarios" element={<SimulateDemonstation />} />
        <Route
            path="citizen"
            element={
              <CitizenLayout />
            }
          >
            <Route index element={<CitizenAroundMe />} />

            <Route path="around-me" element={<CitizenAroundMe />} />
            <Route path="traffic" element={<CitizenTraffic />} />
            <Route path="alerts" element={<CitizenAlerts />} />
            <Route path="services" element={<CitizenServices />} />
            <Route path="assistant" element={<CitizenAssistant />} />
            <Route path="*" element={<Navigate to="/citizen" replace />} />
          </Route>
        <Route path="/" element={<MainLayout />}>
          {/* Global Pages */}
          <Route index element={<UnifiedDashboard />} />
          <Route path="launchpad" element={<DemonstrationInput />} />
          <Route path="digital-twin" element={<DigitalTwinMap />} />

          {/* Department: Traffic */}
          <Route
            path="traffic"
            element={
              <DepartmentLayout
                title="Traffic"
                sidebarConfig={departmentSidebars.traffic}
              />
            }
          >
            <Route index element={<TrafficOverview />} />
            <Route path="cameras" element={<TrafficCameras />} />
            <Route path="live" element={<LiveTraffic />} />
            <Route path="route-intelligence" element={<RouteIntelligence />} />
            <Route path="signals" element={<TrafficSignals />} />
            <Route path="emergency-priority" element={<EmergencyPriority />} />
            <Route path="predictions" element={<AIPredictions />} />
            <Route path="congestion" element={<CongestionHeatmap />} />
            <Route
              path="road-closures"
              element={<PlaceholderPage title="Road Closures" />}
            />

            <Route path="*" element={<Navigate to="/traffic" replace />} />
          </Route>

          {/* Department: Hospital */}
          <Route
            path="hospital"
            element={
              <DepartmentLayout
                title="Hospital Operations"
                sidebarConfig={departmentSidebars.hospital}
              />
            }
          >
            <Route index element={<HospitalOverview />} />
            <Route path="hospitals" element={<Hospitals />} />
            <Route
              path="emergency-ambulance"
              element={<EmergencyAmbulance />}
            />
            <Route path="bed-availability" element={<BedAvailability />} />
            <Route
              path="emergency-department"
              element={<EmergencyDepartment />}
            />
            <Route path="patient-flow" element={<PatientFlow />} />
            <Route path="capacity" element={<HospitalCapacity />} />
            <Route
              path="resource-management"
              element={<ResourceManagement />}
            />
            <Route path="agent" element={<HospitalAgent />} />

            <Route path="*" element={<Navigate to="/hospital" replace />} />
          </Route>
          {/* Department: Police */}
          <Route
            path="police"
            element={
              <DepartmentLayout
                title="Police Operations"
                sidebarConfig={departmentSidebars.police}
              />
            }
          >
            <Route index element={<PoliceOverview />} />
            <Route path="incidents" element={<PoliceIncidents />} />
            <Route path="crime-monitoring" element={<CrimeMonitoring />} />
            <Route path="cameras" element={<TrafficCameras />} />
            <Route path="cctv" element={<CCTVIntelligence />} />
            <Route
              path="emergency-response"
              element={<PoliceEmergencyResponse />}
            />
            <Route path="agent" element={<PoliceAgent />} />
            <Route path="*" element={<Navigate to="/police" replace />} />
          </Route>
          {/* Department: Fire */}
          <Route
            path="fire"
            element={
              <DepartmentLayout
                title="Fire Operations"
                sidebarConfig={departmentSidebars.fire}
              />
            }
          >
            <Route index element={<FireOverview />} />
            <Route path="incidents" element={<FireIncidents />} />
            <Route path="dispatch" element={<FireResponseDispatch />} />
            <Route path="stations" element={<FireStations />} />
            <Route path="hydrants" element={<FireHydrants />} />
            <Route path="agent" element={<FireAgent />} />
            <Route path="analytics" element={<FireAnalytics />} />
            <Route path="*" element={<Navigate to="/fire" replace />} />
          </Route>
        </Route>
      </Routes>
    </Router>
  );
}
