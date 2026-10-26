import React, { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import useIncidentDetail from "../hooks/useIncidentDetail";
import { SeverityBadge, StatusBadge } from "../components/badges";
import { DetailsSkeleton, ErrorState } from "../components/states";
import IncidentMap from "./IncidentMap";
import IncidentTimeline from "./IncidentTimeline";
import IncidentOverviewCard from "./IncidentOverviewCard";
import PoliceResponseCard from "./PoliceResponseCard";
import DepartmentCoordination from "./DepartmentCoordination";
import IncidentAIPanel from "./IncidentAIPanel";
import IncidentActions from "./IncidentActions";
import RelatedIncidents from "./RelatedIncidents";

const BackButton = ({ onClick }) => (
  <button type="button" onClick={onClick} className="text-[11px] font-bold uppercase tracking-wider px-3 py-1.5 rounded border border-slate-600 text-slate-300 hover:bg-slate-800">
    ← All Incidents
  </button>
);

export default function IncidentDetails({ id, liveSignature }) {
  const navigate = useNavigate();
  const { data, loading, notFound, error, refresh } = useIncidentDetail(id, liveSignature);
  const [recenterKey, setRecenterKey] = useState(0);

  const goBack = useCallback(() => navigate("/police/incidents"), [navigate]);
  const viewLocation = useCallback(() => {
    document.getElementById("incident-map")?.scrollIntoView({ behavior: "smooth", block: "center" });
    setRecenterKey((k) => k + 1);
  }, []);

  if (loading) return <DetailsSkeleton />;

  if (notFound) {
    return (
      <div className="space-y-4">
        <BackButton onClick={goBack} />
        <ErrorState onRetry={goBack} title="Incident not found" message="This incident does not exist or is no longer available." />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="space-y-4">
        <BackButton onClick={goBack} />
        <ErrorState onRetry={refresh} />
      </div>
    );
  }

  const { incident, related, stations, crimes } = data;

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-700 pb-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <SeverityBadge severity={incident.severityKey} />
            <StatusBadge status={incident.statusKey} />
          </div>
          <h1 className="text-xl sm:text-2xl font-bold uppercase tracking-wide text-slate-100 truncate">{incident.typeLabel}</h1>
          <div className="text-xs text-slate-400 truncate">
            {incident.location} · <span className="font-mono text-blue-400">{incident.id}</span>
          </div>
        </div>
        <BackButton onClick={goBack} />
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-8 flex flex-col gap-5">
          <IncidentMap incident={incident} stations={stations} crimes={crimes} recenterKey={recenterKey} />
          <IncidentTimeline incident={incident} />
        </div>

        <div className="lg:col-span-4 flex flex-col gap-5">
          <IncidentOverviewCard incident={incident} />
          <PoliceResponseCard incident={incident} />
          <DepartmentCoordination departments={incident.departments} />
          <IncidentAIPanel incident={incident} />
          <IncidentActions incident={incident} onChanged={refresh} onViewLocation={viewLocation} />
        </div>
      </div>

      <RelatedIncidents current={incident} items={related} />
    </div>
  );
}
