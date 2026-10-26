export const mainSidebar = [
  {
    title: "Dashboard",
    path: "/",
    icon: "dashboard",
  },
  {
    title: "Simulation Launchpad",
    path: "/launchpad",
    icon: "launchpad",
  },
  {
    title: "Digital Twin",
    path: "/digital-twin",
    icon: "digital-twin",
  },
  {
    title: "Traffic Department",
    path: "/traffic",
    icon: "traffic",
  },
  {
    title: "Hospital Department",
    path: "/hospital",
    icon: "hospital",
  },
  {
    title: "Police Department",
    path: "/police",
    icon: "police",
  },
  {
    title: "Fire Department",
    path: "/fire",
    icon: "fire",
  },
  // {
  //   title: "Utility Department",
  //   path: "/utility",
  //   icon: "utility",
  // },
  {
    title: "Citizen Services",
    path: "/citizen",
    icon: "citizen",
  },
  // {
  //   title: "City Reports",
  //   path: "/reports",
  //   icon: "reports",
  // },
  // {
  //   title: "AI Analytics",
  //   path: "/analytics",
  //   icon: "analytics",
  // },
  // {
  //   title: "System Settings",
  //   path: "/settings",
  //   icon: "settings",
  // },
];

export const departmentSidebars = {
  //------------------------------------------------
  // TRAFFIC
  //------------------------------------------------
  traffic: [
    { title: "Overview", path: "/traffic" },
    { title: "Traffic Cameras", path: "/traffic/cameras" },
    { title: "Live Traffic", path: "/traffic/live" },
    { title: "Route Intelligence", path: "/traffic/route-intelligence" },
    { title: "Traffic Signals", path: "/traffic/signals" },
    // { title: "Emergency Priority", path: "/traffic/emergency-priority" },
    // { title: "AI Predictions", path: "/traffic/predictions" },
    // { title: "Congestion & Density", path: "/traffic/congestion" },
    // { title: "Road Closures", path: "/traffic/road-closures" },
  ],
  //------------------------------------------------
  // HOSPITAL
  //------------------------------------------------
  hospital: [
    { title: "Overview", path: "/hospital" },
    { title: "Hospitals", path: "/hospital/hospitals" },
    { title: "Emergency & Ambulance", path: "/hospital/emergency-ambulance" },
    { title: "Bed Availability", path: "/hospital/bed-availability" },
    { title: "Emergency Department", path: "/hospital/emergency-department" },
    // { title: "Patient Flow", path: "/hospital/patient-flow" },
    // { title: "Hospital Capacity", path: "/hospital/capacity" },
    // { title: "Resource Management", path: "/hospital/resource-management" },
    { title: "Hospital Agent", path: "/hospital/agent" },
  ],

  //------------------------------------------------
  // POLICE
  //------------------------------------------------
  police: [
    { title: "Overview", path: "/police" },
    { title: "Incidents", path: "/police/incidents" },
    { title: "Crime Monitoring", path: "/police/crime-monitoring" },
    { title: "Cctv Monitoring", path: "/police/cameras" },
    { title: "CCTV Intelligence", path: "/police/cctv" },
    // { title: "Emergency Response", path: "/police/emergency-response" },
    { title: "Police Agent", path: "/police/agent" },
  ],

  //------------------------------------------------
  // FIRE
  //------------------------------------------------
fire: [
  // { title: "Overview", path: "/fire" },
  { title: "Fire Incidents", path: "/fire/incidents" },
  { title: "Response & Dispatch", path: "/fire/dispatch" },
  { title: "Fire Stations", path: "/fire/stations" },
  // { title: "Hydrant & Water Supply", path: "/fire/hydrants" },
  { title: "Fire Agent", path: "/fire/agent" },
  // { title: "Fire Analytics", path: "/fire/analytics" },
],
  //------------------------------------------------
  // UTILITY
  //------------------------------------------------
  // utility: [
  //   { title: "Overview", path: "/utility" },
  //   { title: "Electricity", path: "/utility/electricity" },
  //   { title: "Water Supply", path: "/utility/water" },
  //   { title: "Gas Network", path: "/utility/gas" },
  //   { title: "Waste Management", path: "/utility/waste" },
  //   { title: "Street Lights", path: "/utility/lights" },
  //   { title: "Sewage System", path: "/utility/sewage" },
  //   { title: "Power Grid", path: "/utility/grid" },
  //   { title: "Water Quality", path: "/utility/quality" },
  //   { title: "Maintenance", path: "/utility/maintenance" },
  //   { title: "Utility Analytics", path: "/utility/analytics" },
  //   { title: "AI Resource Optimization", path: "/utility/predictions" },
  // ],

  //------------------------------------------------
  // CITIZEN
  //------------------------------------------------
citizen: [
  { title: "Around Me", path: "/citizen/around-me" },
  { title: "Traffic", path: "/citizen/traffic" },
  { title: "Alerts", path: "/citizen/alerts" },
  { title: "City Services", path: "/citizen/services" },
  { title: "AI City Assistant", path: "/citizen/assistant" },
],
  //------------------------------------------------
  // REPORTS
  //------------------------------------------------
  reports: [
    { title: "Overview", path: "/reports" },
    { title: "Traffic Reports", path: "/reports/traffic" },
    { title: "Hospital Reports", path: "/reports/hospital" },
    { title: "Police Reports", path: "/reports/police" },
    { title: "Fire Reports", path: "/reports/fire" },
    { title: "Utility Reports", path: "/reports/utility" },
    { title: "Citizen Reports", path: "/reports/citizen" },
    { title: "Download Reports", path: "/reports/download" },
  ],

  //------------------------------------------------
  // ANALYTICS
  //------------------------------------------------
  analytics: [
    { title: "Overview", path: "/analytics" },
    { title: "City KPIs", path: "/analytics/kpi" },
    { title: "Predictive AI", path: "/analytics/predictions" },
    { title: "Resource Allocation", path: "/analytics/resources" },
    { title: "AI Agents", path: "/analytics/agents" },
    { title: "Decision Intelligence", path: "/analytics/decision" },
    { title: "Historical Trends", path: "/analytics/history" },
    { title: "Forecasting", path: "/analytics/forecast" },
  ],

  //------------------------------------------------
  // SETTINGS
  //------------------------------------------------
  settings: [
    { title: "General", path: "/settings" },
    { title: "Users", path: "/settings/users" },
    { title: "Roles & Permissions", path: "/settings/roles" },
    { title: "Departments", path: "/settings/departments" },
    { title: "AI Models", path: "/settings/ai-models" },
    { title: "Notifications", path: "/settings/notifications" },
    { title: "Audit Logs", path: "/settings/logs" },
    { title: "System Backup", path: "/settings/backup" },
  ],
};
