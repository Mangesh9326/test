import { useCallback, useEffect, useRef, useState } from "react";
import { fetchIncidentDetail } from "../api";
import { normalizeIncident } from "../utils";

export default function useIncidentDetail(id, liveSignature, pollMs = 15000) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState(null);
  const activeId = useRef(id);

  const load = useCallback(async () => {
    const requested = id;
    try {
      const res = await fetchIncidentDetail(requested);
      if (activeId.current !== requested) return; // user navigated away
      setData({
        incident: normalizeIncident(res.incident),
        related: (res.related || []).map(normalizeIncident),
        stations: res.nearbyStations || [],
        crimes: res.nearbyCrimes || [],
      });
      setNotFound(false);
      setError(null);
    } catch (e) {
      if (activeId.current !== requested) return;
      if (e.status === 404) setNotFound(true);
      else { console.error("[INCIDENT DETAIL] load failed:", e); setError(e); }
    } finally {
      if (activeId.current === requested) setLoading(false);
    }
  }, [id]);

  // New id: reset and load.
  useEffect(() => {
    activeId.current = id;
    setData(null); setLoading(true); setNotFound(false); setError(null);
    load();
    const t = setInterval(load, pollMs);
    return () => clearInterval(t);
  }, [id, load, pollMs]);

  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    load();
  }, [liveSignature, load]);

  return { data, loading, notFound, error, refresh: load };
}
