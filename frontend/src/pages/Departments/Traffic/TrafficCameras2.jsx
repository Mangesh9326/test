import { useCallback, useEffect, useMemo, useState } from "react";
import { Camera } from "lucide-react";
import TrafficCameraMap from "../../../components/DigitalTwin/TrafficCameraMap";
import { useMapStore } from "../../../store/useMapStore";
import CameraFeedCard from "./components/camera/CameraFeedCard";
import CameraList from "./components/camera/CameraList";
import FullscreenCameraModal from "./components/camera/FullscreenCameraModal";
import TrafficCameraControls from "./components/camera/TrafficCameraControls";
import { buildCameras } from "./components/camera/trafficCameraUtils";

const FALLBACK_VIDEO = "http://localhost:5000/videos/1.mp4";
const ITEMS_PER_PAGE = 9;

export default function TrafficCameras2() {
  const { entities = [], initializeMap } = useMapStore();
  const [searchQuery, setSearchQuery] = useState("");
  const [areaFilter, setAreaFilter] = useState("All");
  const [statusFilter, setStatusFilter] = useState("All");
  const [viewMode, setViewMode] = useState("map");
  const [feedSource, setFeedSource] = useState("loop");
  const [selectedCameraId, setSelectedCameraId] = useState(null);
  const [fullscreenCamera, setFullscreenCamera] = useState(null);
  const [externalFeeds, setExternalFeeds] = useState([]);
  const [availableVideos, setAvailableVideos] = useState([FALLBACK_VIDEO]);
  const [currentPage, setCurrentPage] = useState(1);

  // Traffic Agent Monitoring Heartbeat
  const [agentTick, setAgentTick] = useState(0);
  const [lastAnalyzed, setLastAnalyzed] = useState(Date.now());
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const interval = setInterval(() => {
      setAgentTick(t => t + 1);
      setLastAnalyzed(Date.now());
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const ticker = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(ticker);
  }, []);

  const secondsAgo = Math.floor((now - lastAnalyzed) / 1000);
  const timeAgo = secondsAgo <= 0 ? "just now" : `${secondsAgo}s ago`;

  useEffect(() => {
    if (entities.length) return;
    fetch("/api/map/data")
      .then((response) =>
        response.ok
          ? response.json()
          : Promise.reject(new Error("Unable to load map data")),
      )
      .then(initializeMap)
      .catch((error) => console.error("Failed to fetch map data", error));
  }, [entities.length, initializeMap]);

  useEffect(() => {
    let active = true;
    fetch("/api/videos/list")
      .then((response) =>
        response.ok
          ? response.json()
          : Promise.reject(new Error("Unable to load videos")),
      )
      .then((fileNames) => {
        if (active && Array.isArray(fileNames) && fileNames.length)
          setAvailableVideos(
            fileNames.map(
              (name) =>
                `http://localhost:5000/videos/list/${encodeURIComponent(name)}`,
            ),
          );
      })
      .catch((error) => console.error("Failed to load video list", error));
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (feedSource !== "api") return undefined;
    let active = true;
    const loadExternalFeeds = () =>
      fetch("https://api.data.gov.sg/v1/transport/traffic-images")
        .then((response) =>
          response.ok
            ? response.json()
            : Promise.reject(new Error("Unable to load snapshots")),
        )
        .then((data) => {
          if (active) setExternalFeeds(data.items?.[0]?.cameras ?? []);
        })
        .catch((error) =>
          console.error("Failed to fetch external CCTV API", error),
        );
    loadExternalFeeds();
    const intervalId = window.setInterval(loadExternalFeeds, 60_000);
    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, [feedSource]);

  const cameras = useMemo(
    () =>
      buildCameras({ entities, feedSource, availableVideos, externalFeeds }),
    [entities, feedSource, availableVideos, externalFeeds],
  );

  const selectedCamera = useMemo(
    () =>
      cameras.find(({ camera_id }) => camera_id === selectedCameraId) ?? null,
    [cameras, selectedCameraId],
  );

  const stats = useMemo(() => {
    const onlineCount = cameras.filter(({ status }) => status === "Online").length;
    return {
      total: cameras.length,
      online: onlineCount,
      offline: cameras.filter(({ status }) => status === "Offline").length,
      monitored: onlineCount // Traffic Agent processes all online streams
    };
  }, [cameras]);

  const areas = useMemo(
    () => [
      "All",
      ...Array.from(
        new Set(cameras.map(({ location_name }) => location_name)),
      ).sort(),
    ],
    [cameras],
  );

  const filteredCameras = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    return cameras.filter((camera) => {
      const matchesSearch =
        !query ||
        [
          camera.camera_code,
          camera.name,
          camera.location_name,
          camera.road_name,
        ].some((value) => value.toLowerCase().includes(query));
      return (
        matchesSearch &&
        (areaFilter === "All" || camera.location_name === areaFilter) &&
        (statusFilter === "All" || camera.status === statusFilter)
      );
    });
  }, [cameras, searchQuery, areaFilter, statusFilter]);

  const totalPages = Math.max(
    1,
    Math.ceil(filteredCameras.length / ITEMS_PER_PAGE),
  );
  const page = Math.min(currentPage, totalPages);
  
  const paginatedCameras = useMemo(
    () =>
      filteredCameras.slice((page - 1) * ITEMS_PER_PAGE, page * ITEMS_PER_PAGE),
    [filteredCameras, page],
  );

  const selectCamera = useCallback(
    (camera) => setSelectedCameraId(camera.camera_id),
    [],
  );
  const openCamera = useCallback((camera) => {
    setSelectedCameraId(camera.camera_id);
    setViewMode("map");
  }, []);
  const resetPage = useCallback(
    (setter) => (value) => {
      setter(value);
      setCurrentPage(1);
    },
    [],
  );

  return (
    <div className="traffic-camera-page relative flex h-full flex-col gap-4 overflow-hidden text-slate-100">
      <TrafficCameraControls
        viewMode={viewMode}
        setViewMode={resetPage(setViewMode)}
        feedSource={feedSource}
        setFeedSource={setFeedSource}
        searchQuery={searchQuery}
        setSearchQuery={resetPage(setSearchQuery)}
        areaFilter={areaFilter}
        setAreaFilter={resetPage(setAreaFilter)}
        statusFilter={statusFilter}
        setStatusFilter={resetPage(setStatusFilter)}
        areas={areas}
        stats={stats}
        timeAgo={timeAgo}
      />
      {viewMode === "map" ? (
        <section className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-12">
          <aside className="custom-scrollbar flex min-h-0 flex-col gap-2 overflow-y-auto pr-1 lg:col-span-4 xl:col-span-3">
            <CameraList
              cameras={filteredCameras}
              selectedCameraId={selectedCameraId}
              onSelect={setSelectedCameraId}
            />
          </aside>
          <div className="flex min-h-0 flex-col gap-4 lg:col-span-8 xl:col-span-9">
            <div className="min-h-52 h-[44%] shrink-0 overflow-hidden rounded-2xl border border-slate-700/80 bg-slate-950 shadow-xl shadow-slate-950/30">
              {selectedCamera ? (
                <CameraFeedCard
                  camera={selectedCamera}
                  onSelect={selectCamera}
                  onExpand={setFullscreenCamera}
                  priority
                  agentTick={agentTick}
                  timeAgo={timeAgo}
                />
              ) : (
                <div className="flex h-full flex-col items-center justify-center bg-[radial-gradient(circle_at_center,rgba(30,41,59,.9),rgba(2,6,23,1))] px-6 text-center">
                  <div className="rounded-2xl border border-slate-700 bg-slate-800/60 p-4 shadow-lg">
                    <Camera className="h-8 w-8 text-blue-300" />
                  </div>
                  <h2 className="mt-4 font-semibold text-slate-200">
                    Select a camera to begin
                  </h2>
                  <p className="mt-1 max-w-md text-sm text-slate-500">
                    Choose a camera from the list or an interactive map marker
                    to inspect its feed.
                  </p>
                </div>
              )}
            </div>
            <div className="min-h-0 flex-1 overflow-hidden rounded-2xl border border-slate-700/80 bg-slate-900 shadow-lg">
              <TrafficCameraMap
                cameras={filteredCameras}
                selectedCamera={selectedCamera}
                onCameraSelect={selectCamera}
              />
            </div>
          </div>
        </section>
      ) : (
        <section className="flex min-h-0 flex-1 flex-col gap-4">
          {paginatedCameras.length ? (
            <div className="custom-scrollbar grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-y-auto pr-1 pb-1 md:grid-cols-2 xl:grid-cols-3">
              {paginatedCameras.map((camera) => (
                <div
                  key={camera.camera_id}
                  className="min-h-60 overflow-hidden rounded-2xl border border-slate-700/70 bg-slate-950 shadow-lg transition duration-300 hover:border-blue-500/60 hover:shadow-blue-950/25"
                >
                  <CameraFeedCard
                    camera={camera}
                    onSelect={openCamera}
                    onExpand={setFullscreenCamera}
                    agentTick={agentTick}
                    timeAgo={timeAgo}
                  />
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center rounded-2xl border border-dashed border-slate-700 text-center text-slate-500">
              <Camera className="mb-3 h-8 w-8" />
              No cameras match these filters.
            </div>
          )}
          {totalPages > 1 && (
            <nav
              className="flex shrink-0 items-center justify-center gap-3 rounded-xl border border-slate-700/60 bg-slate-900/75 p-3"
              aria-label="Camera pages"
            >
              <button
                onClick={() =>
                  setCurrentPage((current) => Math.max(1, current - 1))
                }
                disabled={page === 1}
                className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Previous
              </button>
              <span className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-xs text-slate-400">
                {page} / {totalPages}
              </span>
              <button
                onClick={() =>
                  setCurrentPage((current) => Math.min(totalPages, current + 1))
                }
                disabled={page === totalPages}
                className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Next
              </button>
            </nav>
          )}
        </section>
      )}
      <FullscreenCameraModal
        camera={fullscreenCamera}
        onClose={() => setFullscreenCamera(null)}
        agentTick={agentTick}
        timeAgo={timeAgo}
      />
    </div>
  );
}