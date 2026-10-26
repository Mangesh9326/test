import React from "react";

const pageWindow = (page, count) => {
  const pages = new Set([1, count, page - 1, page, page + 1]);
  const sorted = [...pages].filter((p) => p >= 1 && p <= count).sort((a, b) => a - b);
  const out = [];
  sorted.forEach((p, i) => {
    if (i > 0 && p - sorted[i - 1] > 1) out.push("…");
    out.push(p);
  });
  return out;
};

const btn = "min-w-[2rem] px-2 py-1 text-[11px] font-mono rounded border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed";

export default function IncidentPagination({ page, pageCount, total, pageSize, onPage }) {
  if (total === 0) return null;
  const start = (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, total);

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
      <div className="text-[11px] font-mono text-slate-400">
        Showing {start}–{end} of {total} incidents
      </div>
      {pageCount > 1 && (
        <nav className="flex items-center gap-1 flex-wrap justify-center" aria-label="Pagination">
          <button type="button" className={btn} disabled={page === 1} onClick={() => onPage(page - 1)}>Previous</button>
          {pageWindow(page, pageCount).map((p, i) =>
            p === "…" ? (
              <span key={`gap-${i}`} className="px-1 text-slate-600">…</span>
            ) : (
              <button
                key={p}
                type="button"
                onClick={() => onPage(p)}
                aria-current={p === page ? "page" : undefined}
                className={`${btn} ${p === page ? "bg-blue-500/20 border-blue-500/50 text-blue-200" : ""}`}
              >
                {p}
              </button>
            ),
          )}
          <button type="button" className={btn} disabled={page === pageCount} onClick={() => onPage(page + 1)}>Next</button>
        </nav>
      )}
    </div>
  );
}
