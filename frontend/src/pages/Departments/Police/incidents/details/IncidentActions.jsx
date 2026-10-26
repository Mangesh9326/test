import React, { useState } from "react";
import Panel from "../components/Panel";
import { addTimelineEvent, updateIncidentStatus } from "../api";
import { STATUS_OPTIONS, TIMELINE_EVENT_OPTIONS } from "../constants";
import { titleCase } from "../utils";

const field = "bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-blue-500 w-full";
const primary = "text-[11px] font-bold uppercase tracking-wider px-3 py-1.5 rounded border border-blue-500/50 text-blue-200 bg-blue-500/15 hover:bg-blue-500/25 disabled:opacity-50 disabled:cursor-not-allowed";

/**
 * Only actions backed by real endpoints are shown:
 *   PATCH /api/police/incidents/:id/status
 *   POST  /api/police/incidents/:id/timeline
 * The UI is refreshed from the server afterwards (no optimistic fake success).
 */
export default function IncidentActions({ incident, onChanged, onViewLocation }) {
  const [nextStatus, setNextStatus] = useState("");
  const [eventType, setEventType] = useState(TIMELINE_EVENT_OPTIONS[0].value);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);

  const statusChoices = STATUS_OPTIONS.filter((s) => s !== incident.statusKey);

  const run = async (fn, successText, reset) => {
    setBusy(true);
    setMessage(null);
    try {
      await fn();
      await onChanged();
      reset();
      setMessage({ ok: true, text: successText });
    } catch (e) {
      console.error("[INCIDENT ACTION] failed:", e);
      setMessage({ ok: false, text: e.message || "Action failed." });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel title="Incident Actions">
      <div className="p-4 space-y-4">
        <div>
          <div className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Update Status</div>
          <div className="flex gap-2">
            <select value={nextStatus} onChange={(e) => setNextStatus(e.target.value)} className={field} disabled={busy}>
              <option value="">Select status…</option>
              {statusChoices.map((s) => <option key={s} value={s}>{titleCase(s)}</option>)}
            </select>
            <button
              type="button"
              className={primary}
              disabled={busy || !nextStatus}
              onClick={() => run(() => updateIncidentStatus(incident.id, nextStatus), `Status updated to ${titleCase(nextStatus)}.`, () => setNextStatus(""))}
            >
              Apply
            </button>
          </div>
        </div>

        <div>
          <div className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Add Timeline Event</div>
          <select value={eventType} onChange={(e) => setEventType(e.target.value)} className={`${field} mb-2`} disabled={busy}>
            {TIMELINE_EVENT_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
          <textarea value={note} onChange={(e) => setNote(e.target.value)} maxLength={500} rows={2} placeholder="Describe what happened…" className={`${field} mb-2 resize-none`} disabled={busy} />
          <button
            type="button"
            className={primary}
            disabled={busy || !note.trim()}
            onClick={() => run(() => addTimelineEvent(incident.id, eventType, note.trim()), "Timeline event added.", () => setNote(""))}
          >
            Add Event
          </button>
        </div>

        <button type="button" onClick={onViewLocation} className="w-full text-[11px] font-bold uppercase tracking-wider py-1.5 rounded border border-slate-600 text-slate-300 hover:bg-slate-800">
          View Location
        </button>

        {message && (
          <div role="status" className={`text-[11px] rounded px-2.5 py-1.5 border ${message.ok ? "text-emerald-300 bg-emerald-500/10 border-emerald-500/30" : "text-red-300 bg-red-500/10 border-red-500/30"}`}>
            {message.text}
          </div>
        )}
      </div>
    </Panel>
  );
}
