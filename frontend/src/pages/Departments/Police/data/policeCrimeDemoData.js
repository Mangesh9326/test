export const SYNTHETIC_CRIME_DATA = {
  distribution: [
    { category: "Robbery", percentage: 32, count: 42 },
    { category: "Vehicle Theft", percentage: 24, count: 31 },
    { category: "Suspicious Activity", percentage: 18, count: 24 },
    { category: "Burglary", percentage: 14, count: 18 },
    { category: "Assault", percentage: 12, count: 15 },
  ],
  locations: [
    { name: "Bandra West", level: "Critical", trend: "INCREASING" },
    { name: "Andheri East", level: "High", trend: "STABLE" },
    { name: "Kurla", level: "High", trend: "ELEVATED" },
    { name: "Dadar", level: "Elevated", trend: "DECREASING" },
    { name: "Kurla West", level: "Elevated", trend: "STABLE" }
  ],
  recent: [
    { id: "C-9012", category: "Armed Robbery", location: "Bandra West", severity: "CRITICAL", time: "18:37", status: "INVESTIGATING", lat: 19.0560, lng: 72.8320 },
    { id: "C-9013", category: "Vehicle Theft", location: "Dadar East", severity: "HIGH", time: "18:42", status: "REPORTED", lat: 19.0189, lng: 72.8437 },
    { id: "C-9014", category: "Suspicious Activity", location: "Andheri East", severity: "MEDIUM", time: "18:31", status: "MONITORING", lat: 19.1136, lng: 72.8697 },
    { id: "C-9015", category: "Burglary", location: "Kurla", severity: "HIGH", time: "17:15", status: "INVESTIGATING", lat: 19.0728, lng: 72.8826 }
  ]
};