import React from "react";
import Panel, { Field } from "../components/Panel";
import { SeverityBadge, StatusBadge } from "../components/badges";
import { formatDateTime } from "../utils";

export default function IncidentOverviewCard({ incident }) {
  const lastUpdated = incident.lastEvent?.ts || incident.reportedAt;
  const lat = Number(incident.latitude), lng = Number(incident.longitude);

  return (
    <Panel title="Incident Overview">
      <div className="px-4 py-2">
        <Field label="Incident ID" mono>{incident.id}</Field>
        <Field label="Incident Type">{incident.typeLabel}</Field>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Severity"><SeverityBadge severity={incident.severityKey} /></Field>
          <Field label="Status"><StatusBadge status={incident.statusKey} /></Field>
        </div>
        <Field label="Location">{incident.location}</Field>
        {Number.isFinite(lat) && Number.isFinite(lng) && <Field label="Coordinates" mono>{lat.toFixed(5)}, {lng.toFixed(5)}</Field>}
        <Field label="Reported" mono>{formatDateTime(incident.reportedAt)}</Field>
        <Field label="Last Updated" mono>{formatDateTime(lastUpdated)}</Field>
      </div>
    </Panel>
  );
}
