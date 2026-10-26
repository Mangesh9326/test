import React, { useEffect, useState } from "react";
import useDebouncedValue from "../hooks/useDebouncedValue";
import { DATE_FILTERS, DEPARTMENT_FILTERS, SEVERITY_FILTERS, STATUS_FILTERS, TYPE_FILTERS, DEFAULT_FILTERS } from "../constants";

const SORT_OPTIONS = [
  { value: "severity", label: "Severity" },
  { value: "time", label: "Reported time" },
  { value: "status", label: "Status" },
  { value: "type", label: "Incident type" },
  { value: "location", label: "Location" },
];

const fieldCls = "bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-blue-500 w-full";

const Field = ({ label, children, className = "" }) => (
  <label className={`flex flex-col gap-1 ${className}`}>
    <span className="text-[9px] uppercase tracking-widest text-slate-500">{label}</span>
    {children}
  </label>
);

const Select = ({ value, onChange, options }) => (
  <select value={value} onChange={(e) => onChange(e.target.value)} className={fieldCls}>
    {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
  </select>
);

export default function IncidentFilters({ filters, onChange, onClear, sort, onSortChange }) {
  const [searchText, setSearchText] = useState(filters.search);
  const debounced = useDebouncedValue(searchText, 250);

  useEffect(() => {
    if (debounced !== filters.search) onChange({ search: debounced });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debounced]);

  const dirty = Object.keys(DEFAULT_FILTERS).some((k) => (k === "search" ? searchText : filters[k]) !== DEFAULT_FILTERS[k]);

  const typeOptions = [{ value: "all", label: "All" }, ...TYPE_FILTERS.map(({ value, label }) => ({ value, label })), { value: "other", label: "Other" }];

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 space-y-3">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-12 gap-3 items-end">
        <Field label="Search" className="xl:col-span-4">
          <input
            type="search"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="Search incidents, locations, IDs..."
            className={fieldCls}
          />
        </Field>
        <Field label="Severity" className="xl:col-span-2"><Select value={filters.severity} onChange={(v) => onChange({ severity: v })} options={SEVERITY_FILTERS} /></Field>
        <Field label="Status" className="xl:col-span-2"><Select value={filters.status} onChange={(v) => onChange({ status: v })} options={STATUS_FILTERS} /></Field>
        <Field label="Incident Type" className="xl:col-span-2"><Select value={filters.type} onChange={(v) => onChange({ type: v })} options={typeOptions} /></Field>
        <Field label="Department" className="xl:col-span-2"><Select value={filters.department} onChange={(v) => onChange({ department: v })} options={DEPARTMENT_FILTERS} /></Field>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <Field label="Date" className="w-40"><Select value={filters.date} onChange={(v) => onChange({ date: v })} options={DATE_FILTERS} /></Field>
        {filters.date === "custom" && (
          <>
            <Field label="From" className="w-40"><input type="date" value={filters.from} max={filters.to || undefined} onChange={(e) => onChange({ from: e.target.value })} className={fieldCls} /></Field>
            <Field label="To" className="w-40"><input type="date" value={filters.to} min={filters.from || undefined} onChange={(e) => onChange({ to: e.target.value })} className={fieldCls} /></Field>
          </>
        )}
        <Field label="Sort by" className="w-40 lg:hidden"><Select value={sort.key} onChange={(v) => onSortChange(v)} options={SORT_OPTIONS} /></Field>

        <button
          type="button"
          onClick={() => { setSearchText(""); onClear(); }}
          disabled={!dirty}
          className="ml-auto text-[11px] font-bold uppercase tracking-wider px-3 py-1.5 rounded border border-slate-600 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Clear Filters
        </button>
      </div>
    </div>
  );
}
