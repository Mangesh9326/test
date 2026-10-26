import { useEffect, useMemo } from "react";
// Adjust to your project layout (src/store/usePoliceStore).
import { usePoliceStore } from "../../../../../store/usePoliceStore";

/**
 * Keeps the existing socket/store subscription alive for this page and returns
 * a string that changes whenever the store sees an incident being created,
 * changed or resolved. Data hooks refetch when it changes.
 */
export default function usePoliceLiveSignal() {
  const { activeIncidents, fetchInitialData, subscribeToSockets, unsubscribeFromSockets } = usePoliceStore();

  useEffect(() => {
    try {
      Promise.resolve(fetchInitialData?.()).catch((e) => console.error("[INCIDENTS] store init failed:", e));
    } catch (e) {
      console.error("[INCIDENTS] store init failed:", e);
    }
    subscribeToSockets?.();
    return () => unsubscribeFromSockets?.();
  }, [fetchInitialData, subscribeToSockets, unsubscribeFromSockets]);

  return useMemo(
    () => (activeIncidents || []).map((i) => `${i.incident_id}:${i.status}:${i.severity}`).join("|"),
    [activeIncidents],
  );
}
