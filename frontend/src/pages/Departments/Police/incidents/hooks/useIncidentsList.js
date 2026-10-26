import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchIncidentsAll } from "../api";
import { normalizeIncident } from "../utils";

export default function useIncidentsList(liveSignature, pollMs = 15000) {
  const [raw, setRaw] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const mounted = useRef(true);
  const inFlight = useRef(false);

  const load = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setRefreshing(true);
    try {
      const data = await fetchIncidentsAll();
      if (!mounted.current) return;
      setRaw(Array.isArray(data.incidents) ? data.incidents : []);
      setError(null);
      setLastUpdated(new Date());
    } catch (e) {
      console.error("[INCIDENTS] load failed:", e);
      if (mounted.current) setError(e);
    } finally {
      inFlight.current = false;
      if (mounted.current) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    load();
    const t = setInterval(load, pollMs);
    return () => { mounted.current = false; clearInterval(t); };
  }, [load, pollMs]);

  // Socket-driven refresh (skip the initial value).
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    load();
  }, [liveSignature, load]);

  const incidents = useMemo(() => raw.map(normalizeIncident), [raw]);
  return { incidents, loading, refreshing, error, lastUpdated, refresh: load };
}
