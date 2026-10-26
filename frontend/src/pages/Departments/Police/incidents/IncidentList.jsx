import React, { useCallback, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import useIncidentsList from "./hooks/useIncidentsList";
import { DEFAULT_FILTERS, PAGE_SIZE } from "./constants";
import { computeKpis, filterIncidents, incidentPath, sortIncidents } from "./utils";
import IncidentsHeader from "./components/IncidentsHeader";
import IncidentKpiCards from "./components/IncidentKpiCards";
import IncidentFilters from "./components/IncidentFilters";
import IncidentTable from "./components/IncidentTable";
import IncidentCardList from "./components/IncidentCardList";
import IncidentPagination from "./components/IncidentPagination";
import IncidentPreviewDrawer from "./components/IncidentPreviewDrawer";
import { EmptyState, ErrorState, KpiSkeleton, NoMatches, TableSkeleton } from "./components/states";

export default function IncidentList({ liveSignature }) {
  const navigate = useNavigate();
  const { incidents, loading, refreshing, error, lastUpdated, refresh } = useIncidentsList(liveSignature);

  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [sort, setSort] = useState({ key: "severity", dir: "asc" });
  const [page, setPage] = useState(1);
  const [previewId, setPreviewId] = useState(null);

  const kpis = useMemo(() => computeKpis(incidents), [incidents]);
  const filtered = useMemo(() => filterIncidents(incidents, filters), [incidents, filters]);
  const sorted = useMemo(() => sortIncidents(filtered, sort), [filtered, sort]);

  const pageCount = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount);
  const rows = useMemo(() => sorted.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE), [sorted, safePage]);

  // Preview is derived from the live list, so it updates if the incident changes.
  const previewIncident = useMemo(() => incidents.find((i) => i.id === previewId) || null, [incidents, previewId]);

  const handleFilterChange = useCallback((patch) => { setFilters((f) => ({ ...f, ...patch })); setPage(1); }, []);
  const handleClear = useCallback(() => { setFilters(DEFAULT_FILTERS); setPage(1); }, []);
  const handleSort = useCallback((key) => {
    setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "asc" }));
    setPage(1);
  }, []);
  const handleSortSelect = useCallback((key) => { setSort({ key, dir: "asc" }); setPage(1); }, []);
  const open = useCallback((id) => navigate(incidentPath(id)), [navigate]);
  const closePreview = useCallback(() => setPreviewId(null), []);

  const initialFailure = error && incidents.length === 0 && !loading;

  return (
    <div className="flex flex-col gap-5">
      <IncidentsHeader total={incidents.length} lastUpdated={lastUpdated} onRefresh={refresh} refreshing={refreshing} />

      {loading ? (
        <>
          <KpiSkeleton />
          <TableSkeleton />
        </>
      ) : initialFailure ? (
        <ErrorState onRetry={refresh} />
      ) : (
        <>
          <IncidentKpiCards kpis={kpis} />

          {error && (
            <div className="text-[11px] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded px-3 py-1.5">
              Live updates are delayed. Showing the most recent data.
            </div>
          )}

          {incidents.length === 0 ? (
            <EmptyState />
          ) : (
            <>
              <IncidentFilters filters={filters} onChange={handleFilterChange} onClear={handleClear} sort={sort} onSortChange={handleSortSelect} />

              {sorted.length === 0 ? (
                <NoMatches onClear={handleClear} />
              ) : (
                <>
                  <IncidentTable rows={rows} sort={sort} onSort={handleSort} onOpen={open} onPreview={setPreviewId} selectedId={previewId} />
                  <IncidentCardList rows={rows} onOpen={open} onPreview={setPreviewId} />
                  <IncidentPagination page={safePage} pageCount={pageCount} total={sorted.length} pageSize={PAGE_SIZE} onPage={setPage} />
                </>
              )}
            </>
          )}
        </>
      )}

      <IncidentPreviewDrawer incident={previewIncident} onClose={closePreview} onOpen={open} />
    </div>
  );
}
