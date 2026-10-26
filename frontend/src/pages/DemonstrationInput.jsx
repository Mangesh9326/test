import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import DigitalTwinMap from "../components/DigitalTwin/MapWidget";

export default function DemonstrationInput() {
  const navigate = useNavigate();

  // Task 4 & 5: Real Mumbai locations
  const mumbaiLocations = [
    {
      id: "CAM-101",
      name: "Ruia College Road",
      road: "Matunga West",
      lat: 19.023845829174906,
      lng: 72.8496884047973,
    },
    {
      id: "CAM-102",
      name: "Dadar Railway Station",
      road: "Dr. Ambedkar Rd",
      lat: 19.018,
      lng: 72.8436,
    },
    {
      id: "CAM-103",
      name: "Wadala Highway (Eastern Freeway Junction)",
      road: "Eastern Freeway",
      lat: 19.0218,
      lng: 72.8745,
    },
  ];

  const [selectedCamera, setSelectedCamera] = useState(mumbaiLocations[0]);
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState("");
  const [scenarioDetails, setScenarioDetails] = useState(null);
  const [customFile, setCustomFile] = useState(null);

  // Execution & Pipeline states
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [isComplete, setIsComplete] = useState(false);

  // Phase 1: State to hold the actual database Incident ID returned from the backend
  const [generatedIncidentId, setGeneratedIncidentId] = useState(null);

  // Multi-Agent activation state
  const [activeAgents, setActiveAgents] = useState([]);
  const [detectionSummary, setDetectionSummary] = useState(null);
  const [replaySummary, setReplaySummary] = useState(null);

  const logContainerRef = useRef(null);
  const videoRef = useRef(null);

  // Enterprise Processing Pipeline Stages
  const processingStages = [
    "Queued",
    "Uploading Video",
    "Extracting Frames",
    "Loading YOLO Model",
    "Object Detection",
    "Object Tracking",
    "Generating Timeline",
    "Saving Database",
    "Preparing Replay",
    "Completed",
  ];

  // Comprehensive Debugging hook to print state changes to the console
  useEffect(() => {
    console.group("[DEBUG CONSOLE] DemonstrationInput State Snapshot");
    console.log("Selected Camera:", selectedCamera);
    console.log("Selected Scenario ID:", selectedScenarioId);
    console.log("Scenario Details:", scenarioDetails);
    console.log("Custom File Selected:", customFile ? customFile.name : "None");
    console.log("Processing Status:", {
      isProcessing,
      progress,
      currentStageIndex,
      isComplete,
    });
    console.log("Active Multi-Agents:", activeAgents);
    console.log("Detection Summary Metrics:", detectionSummary);
    console.log("Generated Incident ID (Phase 1):", generatedIncidentId);
    console.groupEnd();
  }, [
    selectedCamera,
    selectedScenarioId,
    scenarioDetails,
    customFile,
    isProcessing,
    progress,
    isComplete,
    activeAgents,
    detectionSummary,
    generatedIncidentId,
  ]);

  useEffect(() => {
    console.log("[DEBUG] DemonstrationInput: Fetching scenarios...");
    fetch("http://localhost:5000/api/simulation/scenario")
      .then((res) => res.json())
      .then((data) => {
        console.log(
          "[DEBUG] DemonstrationInput: Scenarios fetched successfully:",
          data,
        );
        const formatted = Array.isArray(data) ? data : [];
        formatted.push({
          id: "live",
          name: "Live (Real-Time YOLOv8 Inference)",
          description: "Stream custom CCTV and execute live edge inference.",
        });
        setScenarios(formatted);
        if (formatted.length > 0) {
          setSelectedScenarioId(formatted[0].id);
        }
      })
      .catch((err) => {
        console.warn(
          "[WARNING] Failed to fetch backend scenarios, using default operational fallback.",
          err,
        );
        const fallback = [
          {
            id: 1,
            name: "Heavy Rain Accident",
            description: "Multi-vehicle collision during heavy precipitation.",
          },
          {
            id: 2,
            name: "Building Fire",
            description: "Commercial complex fire emergency response.",
          },
          {
            id: 3,
            name: "Bank Robbery & Pursuit",
            description: "City-wide police pursuit and CCTV tracking.",
          },
          {
            id: "live",
            name: "Live (Real-Time YOLOv8 Inference)",
            description: "Stream custom CCTV and execute live edge inference.",
          },
        ];
        setScenarios(fallback);
        setSelectedScenarioId(1);
      });
  }, []);

  useEffect(() => {
    if (!selectedScenarioId || selectedScenarioId === "live") {
      setScenarioDetails(null);
      return;
    }

    setCustomFile(null);

    console.log(
      `[DEBUG] DemonstrationInput: Fetching scenario details for ID: ${selectedScenarioId}`,
    );
    fetch(`http://localhost:5000/api/simulation/scenario/${selectedScenarioId}`)
      .then((res) => res.json())
      .then((data) => {
        console.log(
          "[DEBUG] DemonstrationInput: Scenario details fetched:",
          data,
        );
        setScenarioDetails(data);
      })
      .catch((err) => {
        console.warn(
          "[WARNING] Could not fetch scenario metadata, using structured fallback.",
        );
        setScenarioDetails({
          name: "Active Simulation Replay",
          description: "Standardized municipal operations scenario.",
          video_file: `${selectedScenarioId}.mp4`,
        });
      });
  }, [selectedScenarioId]);

  // Video Auto-Play logic
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load();
      const playPromise = videoRef.current.play();
      if (playPromise !== undefined) {
        playPromise.catch((error) => {
          if (error.name !== "AbortError") {
            console.warn("Auto-play prevented by browser:", error);
          }
        });
      }
    }
  }, [scenarioDetails, customFile, selectedScenarioId]);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const handleCameraSelect = (cam) => {
    console.log("[DEBUG] Camera selected:", cam);
    setSelectedCamera(cam);
  };

  const handleRunPipeline = async (e) => {
    e.preventDefault();

    if (selectedScenarioId === "live" && !customFile) {
      alert("No video selected for Live inference.");
      return;
    }

    console.log("[DEBUG] DemonstrationInput: Starting Pipeline Execution...");
    setIsProcessing(true);
    setIsComplete(false);
    setCurrentStageIndex(0);
    setProgress(0);
    setLogs([]);
    setActiveAgents([]);
    setDetectionSummary(null);
    setReplaySummary(null);
    setGeneratedIncidentId(null); // Phase 1: Reset ID on new run

    if (selectedScenarioId === "live") {
      executeLivePipeline();
    } else {
      executeReplayPipeline();
    }
  };

  const executeLivePipeline = async () => {
    const formData = new FormData();
    if (customFile) formData.append("video", customFile);
    formData.append("cameraId", selectedCamera.id);
    formData.append("location", selectedCamera.name);
    formData.append("lat", selectedCamera.lat);
    formData.append("lng", selectedCamera.lng);

    try {
      console.log(
        "[DEBUG] DemonstrationInput: Uploading Custom Video payload...",
        {
          file: customFile?.name,
          camera: selectedCamera.id,
        },
      );
      const res = await fetch(
        "http://localhost:5000/api/simulation/live/upload",
        {
          method: "POST",
          body: formData,
        },
      );
      const data = await res.json();
      if (data.incidentId) {
        setGeneratedIncidentId(data.incidentId);
      }
      console.log(
        "[DEBUG] DemonstrationInput: Upload Response received:",
        data,
      );

      const uploadId = data.uploadId;
      pollLiveProcessing(uploadId);
    } catch (err) {
      console.error("[LIVE PIPELINE ERROR]", err);
      setLogs((prev) => [
        ...prev,
        `[ERROR] Live inference pipeline failed: ${err.message}`,
      ]);
      setIsProcessing(false);
    }
  };

  const pollLiveProcessing = (uploadId) => {
    let stage = 0;
    const interval = setInterval(async () => {
      stage++;
      setCurrentStageIndex(Math.min(stage, processingStages.length - 1));

      try {
        const res = await fetch(
          `http://localhost:5000/api/simulation/live/${uploadId}/log`,
        );
        const data = await res.json();
        console.log(
          `[DEBUG] DemonstrationInput: Polling Status (Upload ID: ${uploadId}) ->`,
          data,
        );

        if (data.logs) {
          setLogs(data.logs);
        }

        // Dynamically track progress, ensuring 100% on completion
        if (
          data.status === "Completed" ||
          stage >= processingStages.length - 1
        ) {
          clearInterval(interval);
          setProgress(100);
          setCurrentStageIndex(processingStages.length - 1);

          // Phase 1: Extract actual incident ID from backend response
          const actualIncidentId = data.incidentId || data.summary?.incidentId;
          if (actualIncidentId) {
            console.log(
              "[DEBUG] Phase 1 Sync: Live Incident ID Captured ->",
              actualIncidentId,
            );
            setGeneratedIncidentId(actualIncidentId);
          }

          console.log(
            "[DEBUG] DemonstrationInput: Pipeline completed successfully. Summary:",
            data.summary,
          );
          finalizeSuccessfulProcessing(data.summary || {});
        } else {
          // Progress visually if real progress isnt returned
          setProgress(
            data.progress || (stage / (processingStages.length - 1)) * 100,
          );
        }
      } catch (e) {
        console.error(
          "[DEBUG] DemonstrationInput: Polling execution error:",
          e,
        );
        clearInterval(interval);
        setIsProcessing(false);
      }
    }, 1200);
  };

  const executeReplayPipeline = async () => {
    try {
      console.log(
        `[DEBUG] DemonstrationInput: Executing Simulated Replay for Scenario ID: ${selectedScenarioId}`,
      );

      // FIX: Send location, lat, and lng to the backend so the incident spawns at the correct camera
      const queryParams = new URLSearchParams({
        location: selectedCamera.name,
        lat: selectedCamera.lat,
        lng: selectedCamera.lng,
      }).toString();

      const res = await fetch(
        `http://localhost:5000/api/simulation/scenario/${selectedScenarioId}/log?${queryParams}`,
      );
      const data = await res.json();
      console.log("[DEBUG] DemonstrationInput: Replay Logs Fetched:", data);

      // ... rest of the function stays exactly the same

      // Support for updated backend structure which wraps logs in data.logs
      const rawLogs = Array.isArray(data.logs)
        ? data.logs
        : Array.isArray(data)
          ? data
          : [
              "Loading replay scenario...",
              "Loading AI metadata...",
              "Frame 1: Vehicle detected (Confidence 99%)",
              "Frame 42: Anomaly signature isolated",
              "Timeline synchronized successfully",
            ];

      let stage = 0;
      const timer = setInterval(() => {
        stage++;
        setCurrentStageIndex(Math.min(stage, processingStages.length - 1));
        setProgress((stage / (processingStages.length - 1)) * 100);

        if (stage - 1 < rawLogs.length) {
          setLogs((prev) => [
            ...prev,
            `[${new Date().toLocaleTimeString()}] ${rawLogs[stage - 1]}`,
          ]);
        }

        if (stage >= processingStages.length) {
          clearInterval(timer);
          setProgress(100); // Enforce 100% processing bar completion

          // Use realistic simulated stats based on scenario meta details
          const simEvents = scenarioDetails?.timeline_events_count || 6;
          const simTracking = scenarioDetails?.detection_count || 42;

          const stats = {
            cars: Math.floor(simTracking * 0.5),
            persons: Math.floor(simTracking * 0.3),
            motorcycles: Math.floor(simTracking * 0.1),
            bus: 2,
            truck: 1,
            emergency: simEvents > 0 ? 1 : 0,
            confidence: "97.2%",
            trackingIDs: simTracking,
            timelineEvents: simEvents,
            duration: scenarioDetails?.duration || "02:00",
          };

          // Phase 1: Extract actual incident ID from backend payload for scenarios
          const actualIncidentId =
            data.incidentId || scenarioDetails?.incident_id;
          if (actualIncidentId) {
            console.log(
              "[DEBUG] Phase 1 Sync: Scenario Incident ID Captured ->",
              actualIncidentId,
            );
            setGeneratedIncidentId(actualIncidentId);
          }

          console.log(
            "[DEBUG] DemonstrationInput: Replay finished with simulated stats:",
            stats,
          );
          finalizeSuccessfulProcessing(stats);
        }
      }, 700);
    } catch (err) {
      console.error("[DEBUG] DemonstrationInput: REPLAY ERROR", err);
      setIsProcessing(false);
    }
  };

  const finalizeSuccessfulProcessing = (summaryStats) => {
    console.log(
      "[DEBUG] DemonstrationInput: Finalizing processing. Analyzing Summary Stats:",
      summaryStats,
    );

    setIsProcessing(false);
    setIsComplete(true);
    setProgress(100);

    // Structure the detected objects formatting to handle both Live DB Arrays and Simulation Objects
    let formattedObjects = {
      cars: 0,
      persons: 0,
      bus: 0,
      truck: 0,
      emergency: 0,
      motorcycles: 0,
    };
    let trackingIDs = 0;
    let timelineEvents = 0;

    if (summaryStats && Array.isArray(summaryStats.objects)) {
      // Logic for Live DB fetch: Map backend grouping array
      summaryStats.objects.forEach((obj) => {
        const name = obj.object_detected
          ? obj.object_detected.toLowerCase()
          : "";
        if (name === "car") formattedObjects.cars = parseInt(obj.count, 10);
        else if (name === "person")
          formattedObjects.persons = parseInt(obj.count, 10);
        else if (name === "bus") formattedObjects.bus = parseInt(obj.count, 10);
        else if (name === "truck")
          formattedObjects.truck = parseInt(obj.count, 10);
        else if (name === "motorcycle")
          formattedObjects.motorcycles = parseInt(obj.count, 10);
      });
      trackingIDs = summaryStats.trackingIDs || 0;
      timelineEvents = summaryStats.timelineEvents || 0;
    } else if (summaryStats) {
      // Logic for Replay Engine Objects
      formattedObjects = {
        cars: summaryStats.cars || 14,
        persons: summaryStats.persons || 19,
        bus: summaryStats.bus || 2,
        truck: summaryStats.truck || 1,
        emergency: summaryStats.emergency || 3,
        motorcycles: summaryStats.motorcycles || 5,
      };
      trackingIDs = summaryStats.trackingIDs || 34;
      timelineEvents = summaryStats.timelineEvents || 6;
    }

    setDetectionSummary({
      objects: formattedObjects,
      confidence: summaryStats?.confidence || "94.2%",
      trackingIDs: trackingIDs,
      timelineEvents: timelineEvents,
      duration: summaryStats?.duration || "02:00",
    });

    setReplaySummary({
      replayId: `REP-${Math.floor(Math.random() * 89999 + 10000)}`,
      scenario: scenarioDetails?.name || "Active Live Stream",
      frames: 3600,
      objectsDetected: trackingIDs,
      eventsCount: timelineEvents,
      duration: summaryStats?.duration || "02:00",
      status: "Ready for Dispatch",
      cached: true,
    });

    // Intelligent Agent Activation based on actual AI timeline contexts
    determineAndActivateAgents(timelineEvents, formattedObjects);
  };

  const determineAndActivateAgents = (eventCount, objects) => {
    // 1. Core Coordinator is always required to assess the data
    const agentsToWake = ["Decision Coordinator"];

    // 2. Traffic & Transport Dispatch
    if (
      objects.cars > 0 ||
      objects.bus > 0 ||
      objects.truck > 0 ||
      objects.motorcycles > 0
    ) {
      agentsToWake.push("Traffic Agent");
    }

    // 3. Law Enforcement Dispatch
    if (eventCount > 0) {
      agentsToWake.push("Police Agent");
    }

    // 4. Critical Emergency Responders
    if (eventCount >= 2 || objects.emergency > 0) {
      agentsToWake.push("Hospital Agent");
      agentsToWake.push("Fire Agent");
    }

    // 5. Public / Civilian Notifications
    if (eventCount >= 4 || objects.persons > 30) {
      agentsToWake.push("Citizen Agent");
    }

    console.log(
      "[DEBUG] DemonstrationInput: Deploying specific Multi-Agents based on Context:",
      agentsToWake,
    );

    // 6. Trigger UI animations sequentially
    agentsToWake.forEach((agent, index) => {
      setTimeout(
        () => {
          setActiveAgents((prev) => {
            if (!prev.includes(agent)) return [...prev, agent];
            return prev;
          });
        },
        (index + 1) * 350,
      );
    });
  };

  const getVideoSource = () => {
    if (selectedScenarioId === "live") {
      return customFile ? URL.createObjectURL(customFile) : null;
    }
    if (scenarioDetails && scenarioDetails.video_file) {
      return `http://localhost:5000/videos/${scenarioDetails.video_file}`;
    }
    return `http://localhost:5000/videos/${selectedScenarioId}.mp4`;
  };

  return (
    <div className="flex h-[calc(100vh-64px)] bg-city-900 text-gray-100 p-6 gap-6 overflow-hidden font-sans">
      {/* LEFT COLUMN: Controls, Digital Twin, and Setup */}
      <div
        className={`w-1/2 flex flex-col gap-5 overflow-y-auto pr-2 custom-scrollbar transition-opacity duration-500 ${isProcessing ? "opacity-70 pointer-events-none" : "opacity-100"}`}
      >
        <header className="border-b border-city-700/50 pb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse"></span>
              <span className="text-xs font-mono uppercase tracking-widest text-blue-400">
                Operations Control Center
              </span>
            </div>
            <div className="text-[10px] font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/30 px-2.5 py-1 rounded-md">
              Hardware: Ryzen 5 5500 | GTX 1060 6GB | 16GB RAM
            </div>
          </div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-300 tracking-tight mt-1">
            Simulation Launchpad
          </h1>
          <p className="text-xs text-gray-400 mt-1">
            Select an operational Mumbai corridor, inspect telemetry feeds, and
            initialize autonomous AI multi-agent orchestration.
          </p>
        </header>

        {/* Digital Twin Integration */}
        <div className="glass-panel p-4 flex flex-col gap-3 border-city-700/60 shadow-xl">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-widest text-blue-400">
              Step 1: Select Operational Sector (Mumbai GIS)
            </span>
            <span className="text-[10px] bg-city-800 px-2 py-0.5 rounded text-gray-400 border border-city-700 font-mono">
              EPSG:4326
            </span>
          </div>

          <div className="relative w-full h-52 bg-gray-950 rounded-lg overflow-hidden border border-city-700 shadow-inner">
            {/* demoLocation is intentionally always set here (from selectedCamera).
                 MapWidget derives `isDashboard = !demoLocation` internally, so
                 passing demoLocation suppresses the live/critical incident layer
                 on this map. This panel is for picking a camera location only —
                 it must never surface live incidents. The real-time incident
                 layer (with the pulsing critical icon) only renders on the
                 main Digital Twin dashboard route, where DigitalTwinMap is
                 mounted without a demoLocation prop. */}
            <DigitalTwinMap demoLocation={selectedCamera} />
          </div>

          <div className="grid grid-cols-3 gap-2.5">
            {mumbaiLocations.map((cam) => (
              <button
                key={cam.id}
                type="button"
                onClick={() => handleCameraSelect(cam)}
                className={`p-2.5 rounded-lg border text-left transition-all duration-300 cursor-pointer ${selectedCamera.id === cam.id ? "bg-blue-600/20 border-blue-500 text-blue-300 shadow-[0_0_15px_rgba(59,130,246,0.35)] scale-[1.02]" : "bg-city-800/40 border-city-700/50 text-gray-400 hover:bg-city-700/60"}`}
              >
                <div className="font-bold text-xs font-mono">{cam.id}</div>
                <div className="text-[11px] font-semibold text-gray-200 truncate mt-0.5">
                  {cam.name}
                </div>
                <div className="text-[9px] text-blue-400 font-mono mt-1">
                  Zoom 17 Active
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Dynamic Scenario Feed */}
        <form
          onSubmit={handleRunPipeline}
          className="glass-panel p-4 flex flex-col gap-4 border-city-700/60 shadow-xl"
        >
          <span className="text-xs font-bold uppercase tracking-widest text-teal-400">
            Step 2: Scenario Configuration & Execution Mode
          </span>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-gray-400 uppercase font-bold tracking-wider">
                Active Location
              </label>
              <input
                readOnly
                value={selectedCamera.name}
                className="w-full bg-black/50 border border-city-700/80 rounded-lg p-2 text-xs text-blue-300 font-mono outline-none"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-gray-400 uppercase font-bold tracking-wider">
                Camera ID & Axis
              </label>
              <input
                readOnly
                value={`${selectedCamera.id} (${selectedCamera.road})`}
                className="w-full bg-black/50 border border-city-700/80 rounded-lg p-2 text-xs text-blue-300 font-mono outline-none"
              />
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-gray-400 uppercase font-bold tracking-wider">
              Database Scenario Feed (Dynamic API)
            </label>
            <select
              value={selectedScenarioId}
              onChange={(e) => setSelectedScenarioId(e.target.value)}
              className="w-full bg-black/50 border border-city-700/80 rounded-lg p-2 text-xs text-gray-200 outline-none focus:border-teal-500 font-mono cursor-pointer"
            >
              {scenarios.map((scen) => (
                <option key={scen.id} value={scen.id}>
                  {scen.name} — {scen.description}
                </option>
              ))}
            </select>
          </div>

          {selectedScenarioId === "live" && (
            <div className="flex flex-col gap-2 bg-red-500/10 p-3 rounded-lg border border-red-500/30 animate-in fade-in duration-300">
              <label className="text-[10px] text-red-400 uppercase font-bold tracking-wider flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-red-500 animate-ping"></span>{" "}
                  Upload Live CCTV Video Stream
                </span>
                {!customFile && (
                  <span className="text-amber-300 text-[9px] font-mono">
                    Required for Live Inference
                  </span>
                )}
              </label>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setCustomFile(e.target.files[0])}
                className="w-full text-xs text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-bold file:bg-red-600 file:text-white hover:file:bg-red-500 cursor-pointer"
              />
            </div>
          )}

          <button
            type="submit"
            disabled={selectedScenarioId === "live" && !customFile}
            className="mt-2 bg-gradient-to-r from-blue-600 to-teal-600 hover:from-blue-500 hover:to-teal-500 disabled:from-city-800 disabled:to-city-800 disabled:text-gray-500 text-white font-bold py-3 rounded-lg shadow-lg transition-all text-xs uppercase tracking-widest cursor-pointer disabled:cursor-not-allowed"
          >
            {selectedScenarioId === "live" && !customFile
              ? "No video selected."
              : isProcessing
                ? "Executing Edge Inference Pipeline..."
                : "Run YOLO Pipeline & Sync Engine"}
          </button>
        </form>
      </div>

      {/* RIGHT COLUMN: Video Preview, Terminal, Agent Panels, and Summaries */}
      <div className="w-1/2 flex flex-col gap-4 overflow-y-auto pr-1 custom-scrollbar">
        {/* Enterprise Video Preview Panel */}
        <div className="glass-panel p-4 flex flex-col gap-3 border-teal-500/30 shadow-xl bg-gradient-to-br from-city-800/80 to-city-900 shrink-0">
          <div className="flex justify-between items-center border-b border-city-700 pb-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-widest text-teal-400 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Live
              Edge Feed Preview
            </span>
            <span className="text-[10px] font-mono text-gray-400">
              {selectedScenarioId === "live"
                ? "LOCAL UPLOAD STREAM"
                : "DATABASE REPLAY CACHE"}
            </span>
          </div>

          {/* Video Player */}
          <div className="w-full bg-black rounded-lg overflow-hidden border border-city-700 relative aspect-video flex items-center justify-center">
            {getVideoSource() ? (
              <video
                ref={videoRef}
                className="w-full h-full object-contain"
                controls
                autoPlay
                muted
                loop
              >
                <source src={getVideoSource()} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            ) : (
              <div className="text-gray-600 font-mono text-xs flex flex-col items-center gap-2">
                <svg
                  className="w-8 h-8 opacity-50"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  ></path>
                </svg>
                <span>Awaiting Stream Source...</span>
              </div>
            )}

            {/* HUD Overlay */}
            <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-black/60 backdrop-blur-sm px-2 py-1 rounded border border-city-700">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
              <span className="text-[9px] font-mono text-white tracking-widest uppercase">
                REC
              </span>
            </div>
          </div>

          <div className="text-xs text-gray-300 bg-black/30 p-2.5 rounded border border-city-700/60 italic mt-1">
            "
            {scenarioDetails?.description ||
              "Real-time CCTV edge stream simulation."}
            "
          </div>
        </div>

        {/* AI Processing Terminal & Pipeline Progress */}
        <div className="glass-panel flex flex-col overflow-hidden border-indigo-500/30 shadow-2xl h-64 shrink-0">
          <div className="bg-black/80 p-3.5 border-b border-city-700/50 flex justify-between items-center">
            <div className="flex items-center gap-3">
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-yellow-500"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500"></div>
              </div>
              <span className="text-xs font-bold uppercase tracking-widest text-indigo-300">
                Edge Compute Terminal
              </span>
            </div>
            <span className="text-[10px] font-mono text-cyan-300 bg-cyan-500/10 px-2.5 py-0.5 rounded border border-cyan-500/30">
              Stage: {processingStages[currentStageIndex]}
            </span>
          </div>

          <div
            ref={logContainerRef}
            className="flex-grow bg-black p-4 font-mono text-xs overflow-y-auto space-y-2 leading-relaxed custom-scrollbar"
          >
            {logs.length === 0 ? (
              <div className="text-gray-600 italic">
                system@cityos-core:~$ Ready for pipeline execution...
              </div>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="flex items-start gap-2 text-gray-300">
                  <span className="text-indigo-400 shrink-0">›</span>
                  <span>{log}</span>
                </div>
              ))
            )}
            {isProcessing && (
              <div className="flex items-center gap-2 text-indigo-400 animate-pulse mt-1">
                <span>
                  ⠋ Executing {processingStages[currentStageIndex]}...
                </span>
              </div>
            )}
          </div>

          <div className="p-3 bg-city-900 border-t border-city-700/50">
            <div className="flex justify-between text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1">
              <span>Pipeline Stage Progress ({Math.round(progress)}%)</span>
              <span className="text-indigo-400">
                {processingStages[currentStageIndex]}
              </span>
            </div>
            <div className="w-full bg-black rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-gradient-to-r from-indigo-500 to-teal-400 h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Detection Summary Card */}
        {detectionSummary && (
          <div className="glass-panel p-4 border-blue-500/30 bg-gradient-to-br from-blue-950/30 to-city-900 animate-in fade-in duration-500 shrink-0">
            <div className="flex justify-between items-center mb-3 border-b border-blue-500/20 pb-2">
              <span className="text-xs font-bold uppercase tracking-widest text-blue-400 flex items-center gap-2">
                <svg
                  className="w-4 h-4 text-blue-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
                YOLOv8 Detection Summary Analytics
              </span>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                Confidence: {detectionSummary.confidence}
              </span>
            </div>
            <div className="grid grid-cols-4 gap-2 text-center font-mono">
              <div className="bg-black/40 p-2 rounded border border-city-700">
                <div className="text-[9px] text-gray-500 uppercase">Cars</div>
                <div className="text-sm font-bold text-blue-300">
                  {detectionSummary.objects.cars || 0}
                </div>
              </div>
              <div className="bg-black/40 p-2 rounded border border-city-700">
                <div className="text-[9px] text-gray-500 uppercase">
                  Persons
                </div>
                <div className="text-sm font-bold text-teal-300">
                  {detectionSummary.objects.persons || 0}
                </div>
              </div>
              <div className="bg-black/40 p-2 rounded border border-city-700">
                <div className="text-[9px] text-gray-500 uppercase">
                  Buses/Trucks
                </div>
                <div className="text-sm font-bold text-amber-300">
                  {(detectionSummary.objects.bus || 0) +
                    (detectionSummary.objects.truck || 0)}
                </div>
              </div>
              <div className="bg-black/40 p-2 rounded border border-city-700">
                <div className="text-[9px] text-gray-500 uppercase">
                  Emergency
                </div>
                <div className="text-sm font-bold text-red-400">
                  {detectionSummary.objects.emergency || 0}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Multi-Agent Activation Panel */}
        {activeAgents.length > 0 && (
          <div className="glass-panel p-4 border-emerald-500/30 bg-gradient-to-br from-emerald-950/20 to-city-900 animate-in fade-in duration-500 shrink-0">
            <div className="flex justify-between items-center mb-3 border-b border-emerald-500/20 pb-2">
              <span className="text-xs font-bold uppercase tracking-widest text-emerald-400 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>{" "}
                Multi-Agent System & Decision Coordinator
              </span>
              <span className="text-[10px] font-mono text-gray-400">
                {activeAgents.length} / 8 Active
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 font-mono text-xs">
              {[
                "Traffic Agent",
                "Police Agent",
                "Hospital Agent",
                "Fire Agent",
                "Citizen Agent",
                "Utility Agent",
                "Weather Agent",
                "Decision Coordinator",
              ].map((agentName, idx) => {
                const isActive = activeAgents.includes(agentName);
                return (
                  <div
                    key={idx}
                    className={`flex items-center justify-between p-2 rounded border transition-all ${isActive ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300" : "bg-black/20 border-city-800 text-gray-600"}`}
                  >
                    <span>{agentName}</span>
                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded font-bold ${isActive ? "bg-emerald-500 text-black" : "bg-gray-800 text-gray-500"}`}
                    >
                      {isActive ? "ACTIVE" : "STANDBY"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Replay Summary Card & Dashboard Launch */}
        {isComplete && replaySummary && (
          <div className="glass-panel border-emerald-500/50 p-4 bg-gradient-to-br from-emerald-950/50 to-city-900 flex flex-col gap-3 animate-in fade-in slide-in-from-bottom-4 duration-500 shadow-2xl shrink-0">
            <div className="flex justify-between items-center border-b border-emerald-500/30 pb-2">
              <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M5 13l4 4L19 7"
                  ></path>
                </svg>
                Replay Summary & Ready State Confirmed
              </div>
              <span className="text-[10px] font-mono bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded border border-emerald-500/40">
                {replaySummary.replayId}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 font-mono text-xs">
              <div className="bg-black/50 p-2 rounded border border-city-700">
                <span className="text-[9px] text-gray-500 block uppercase">
                  Scenario
                </span>
                <span className="text-gray-200 font-bold truncate block">
                  {replaySummary.scenario}
                </span>
              </div>
              <div className="bg-black/50 p-2 rounded border border-city-700">
                <span className="text-[9px] text-gray-500 block uppercase">
                  Timeline Events
                </span>
                <span className="text-blue-400 font-bold">
                  {replaySummary.eventsCount} Events
                </span>
              </div>
              <div className="bg-black/50 p-2 rounded border border-city-700">
                <span className="text-[9px] text-gray-500 block uppercase">
                  Database Status
                </span>
                <span className="text-emerald-400 font-bold">
                  Cached & Indexed
                </span>
              </div>
            </div>

            {/* Phase 1: Conditional navigation utilizing the dynamically generated Incident ID */}
            <button
              onClick={() => {
                if (generatedIncidentId) {
                  navigate(`/?incidentId=${generatedIncidentId}`);
                } else {
                  navigate("/");
                }
              }}
              className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold py-3 rounded-lg text-xs uppercase tracking-widest shadow-[0_0_20px_rgba(16,185,129,0.4)] transition-all cursor-pointer flex items-center justify-center gap-2"
            >
              <span>Launch Synchronized Smart City Dashboard</span>
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M14 5l7 7m0 0l-7 7m7-7H3"
                ></path>
              </svg>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
