import React from "react";
import Panel, { Field } from "../components/Panel";
import { formatClock, getPoliceDepartment } from "../utils";

const NONE = <span className="text-slate-500">Not yet recorded</span>;

export default function PoliceResponseCard({ incident }) {
  const police = getPoliceDepartment(incident.departments);

  // Everything below is derived from real department data and timeline events.
  const events = incident.events;
  const unitEvent = [...events].reverse().find((e) => e.unit);
  const dispatched = events.find((e) => /dispatch/i.test(e.label));
  const arrived = events.find((e) => /arriv|on scene/i.test(e.label));
  const assignedUnit = police?.unit || incident.assigned_unit || unitEvent?.unit || null;

  let responseStatus = "Awaiting assignment";
  if (arrived) responseStatus = "On scene";
  else if (dispatched) responseStatus = "Dispatched";
  else if (police) responseStatus = "Assigned";

  return (
    <Panel title="Police Response">
      <div className="px-4 py-2">
        <Field label="Police Role">{police ? police.role || "Responding" : "Not assigned"}</Field>
        <Field label="Assigned Unit" mono>{assignedUnit || NONE}</Field>
        <Field label="Response Status">{responseStatus}</Field>
        <Field label="Arrival Time" mono>{arrived ? formatClock(arrived.ts) : NONE}</Field>
      </div>
    </Panel>
  );
}
