const request = async (url, options = {}) => {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return body;
};

const enc = encodeURIComponent;

export const fetchIncidentsAll = () => request("/api/police/incidents/all");
export const fetchIncidentDetail = (id) => request(`/api/police/incidents/${enc(id)}`);
export const fetchAgentRecommendation = () => request("/api/police/agent/recommendation");

export const updateIncidentStatus = (id, status) =>
  request(`/api/police/incidents/${enc(id)}/status`, { method: "PATCH", body: JSON.stringify({ status }) });

export const addTimelineEvent = (id, event, description) =>
  request(`/api/police/incidents/${enc(id)}/timeline`, { method: "POST", body: JSON.stringify({ event, description }) });
