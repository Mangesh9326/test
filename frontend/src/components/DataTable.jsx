import React from 'react';

export default function DataTable({ headers, data }) {
  return (
    <div className="glass-panel overflow-hidden">
      <table className="w-full text-left text-sm text-gray-300">
        <thead className="text-xs uppercase bg-city-800/50 text-gray-400 border-b border-city-700/50">
          <tr>
            {headers.map((header, idx) => (
              <th key={idx} className="px-4 py-3">{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIdx) => (
            <tr key={rowIdx} className="border-b border-city-700/30 hover:bg-city-800/40 transition-colors">
              {Object.values(row).map((cell, colIdx) => (
                <td key={colIdx} className="px-4 py-3">{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}