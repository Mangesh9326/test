import React from "react";
import { useSearchParams } from "react-router-dom";
import usePoliceLiveSignal from "./hooks/usePoliceLiveSignal";
import IncidentList from "./IncidentList";
import IncidentDetails from "./details/IncidentDetails";

/**
 * Route: /police/incidents            -> incident list
 *        /police/incidents?id=<id>    -> incident details (record fetched by id)
 */
export default function Incidents() {
  const [params] = useSearchParams();
  const id = params.get("id");
  const liveSignature = usePoliceLiveSignal();

  return (
    <div className="p-4 sm:p-6 min-h-screen overflow-y-auto bg-slate-950 font-sans text-slate-100">
      {id ? <IncidentDetails id={id} liveSignature={liveSignature} /> : <IncidentList liveSignature={liveSignature} />}
    </div>
  );
}
