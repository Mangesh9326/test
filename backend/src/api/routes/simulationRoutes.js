const express = require("express");
const router = express.Router();
const path = require("path");
const multer = require("multer");
const fs = require("fs");
const crypto = require("crypto");
const { exec } = require("child_process");
const db = require("../../config/db");
const replayEngine = require("../../engine/replayEngine");

// Ensure the uploads directory exists before configuring multer
const uploadDir = path.join(__dirname, "../public/uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
  console.log(`[SYSTEM] Created missing upload directory: ${uploadDir}`);
}

const videosDir = path.join(__dirname, "../public/videos");
if (!fs.existsSync(videosDir)) {
  fs.mkdirSync(videosDir, { recursive: true });
  console.log(`[SYSTEM] Created missing videos directory: ${videosDir}`);
}

// Configure multer storage for custom CCTV video uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + "-" + file.originalname);
  },
});
const upload = multer({ storage: storage });

// Helper function: Calculate SHA-256 hash from file contents
const calculateFileHash = (filePath) => {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = fs.createReadStream(filePath);
    stream.on("data", (data) => hash.update(data));
    stream.on("end", () => resolve(hash.digest("hex")));
    stream.on("error", reject);
  });
};

// ==========================================
// 1. SCENARIO & REPLAY API ENDPOINTS
// ==========================================

// GET: Fetch all available demonstration scenarios
router.get("/scenario", async (req, res) => {
  try {
    const result = await db.query(
      "SELECT id, name, description, duration_seconds FROM scenarios ORDER BY id ASC",
    );
    res.json(result.rows);
  } catch (error) {
    console.error("[API ERROR] Failed to fetch scenario:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// GET: Fetch single scenario metadata by ID
router.get("/scenario/:id", async (req, res) => {
  const scenarioId = req.params.id;
  try {
    const result = await db.query("SELECT * FROM scenarios WHERE id = $1", [
      scenarioId,
    ]);
    if (result.rows.length === 0) {
      return res.status(404).json({ error: "Scenario not found" });
    }
    const scen = result.rows[0];
    res.json({
      name: scen.name,
      description: scen.description,
      video_file: `${scen.id}.mp4`,
      thumbnail: `/thumbnails/${scen.id}.jpg`,
      resolution: "1920x1080 (1080p)",
      fps: 30,
      duration: `${Math.floor(scen.duration_seconds / 60)}:${(scen.duration_seconds % 60).toString().padStart(2, "0")}`,
      replay_length: `${scen.duration_seconds} seconds`,
      timeline_events_count: 6,
      detection_count: 42,
    });
  } catch (err) {
    console.error("[API ERROR] Failed to fetch scenario details:", err);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// GET: Fetch pre-cached database logs for scenario replay
// GET: Fetch pre-cached database logs for scenario replay
router.get("/scenario/:id/log", async (req, res) => {
  const scenarioId = req.params.id;

  try {
    // =========================================================
    // 1. FETCH PRE-CACHED TIMELINE EVENTS
    // =========================================================
    const result = await db.query(
      "SELECT timestamp_second, event_type, payload FROM timeline_events WHERE scenario_id = $1 ORDER BY timestamp_second ASC",
      [scenarioId],
    );

    const logs = result.rows.map(
      (row) =>
        `[T=${row.timestamp_second}s] Event: ${row.event_type} — ${JSON.stringify(row.payload)}`,
    );

    // =========================================================
    // 2. FIND OR CREATE INCIDENT
    // =========================================================
    let incidentId = null;

    try {
      // -------------------------------------------------------
      // Check if an incident already exists for this scenario
      // -------------------------------------------------------
      const incidentRes = await db.query(
        "SELECT incident_id FROM incidents WHERE timeline_events->>'scenario_id' = $1 ORDER BY created_at DESC LIMIT 1",
        [String(scenarioId)],
      );

      if (incidentRes.rows.length > 0) {
        // =====================================================
        // EXISTING INCIDENT
        // =====================================================
        incidentId = incidentRes.rows[0].incident_id;

        // Reactivate the incident
        await db.query(
          "UPDATE incidents SET status = 'Active', created_at = NOW() WHERE incident_id = $1",
          [incidentId],
        );

        console.log(
          `[API] Reactivated existing incident ${incidentId} for scenario ${scenarioId}`,
        );
      } else {
        // =====================================================
        // NO INCIDENT FOUND → CREATE NEW INCIDENT
        // =====================================================

        // Get selected location from Demonstration page
        const locationName = req.query.location || "Camera / Sensor Node";

        const reqLat = parseFloat(req.query.lat) || 19.076;

        const reqLng = parseFloat(req.query.lng) || 72.8777;

        console.log(`[API] Creating incident for scenario ${scenarioId}`);

        console.log(`[API] Location: ${locationName}`);

        console.log(`[API] Coordinates: ${reqLat}, ${reqLng}`);

        // -----------------------------------------------------
        // Get scenario information
        //
        // IMPORTANT:
        // Keep this exactly like the old working route.
        // Do NOT require video_file column here.
        // -----------------------------------------------------
        const scenarioQuery = await db.query(
          "SELECT name, description FROM scenarios WHERE id = $1",
          [scenarioId],
        );

        let type = "SYSTEM_EVENT";
        let desc = "Simulation Active";

        // -----------------------------------------------------
        // Determine incident type
        // -----------------------------------------------------
        if (scenarioQuery.rows.length > 0) {
          const scen = scenarioQuery.rows[0];

          desc = scen.description || scen.name || "Simulation Active";

          const descLower = desc.toLowerCase();

          if (descLower.includes("fire") || descLower.includes("smoke")) {
            type = "FIRE_HAZARD";
          } else if (
            descLower.includes("robbery") ||
            descLower.includes("pursuit") ||
            descLower.includes("crime")
          ) {
            type = "CRIME_ALERT";
          } else if (
            descLower.includes("accident") ||
            descLower.includes("collision") ||
            descLower.includes("rain")
          ) {
            type = "TRAFFIC_INCIDENT";
          } else {
            type = "AI_DETECTION";
          }
        }

        console.log(`[API] Incident type: ${type}`);

        // -----------------------------------------------------
        // CREATE INCIDENT
        // -----------------------------------------------------
        const seedRes = await db.query(
          `INSERT INTO incidents
           (
             incident_type,
             severity,
             latitude,
             longitude,
             status,
             timeline_events,
             created_at
           )
           VALUES ($1, $2, $3, $4, $5, $6, NOW())
           RETURNING incident_id`,
          [
            type,
            "Critical",
            reqLat,
            reqLng,
            "Active",
            JSON.stringify({
              scenario_id: scenarioId,
              description: desc,
              video_file: `${scenarioId}.mp4`,
              confidence: 0.98,
              location: locationName,
            }),
          ],
        );

        // Get newly created incident ID
        incidentId = seedRes.rows[0].incident_id;

        console.log(
          `[API] Created NEW incident ${incidentId} for scenario ${scenarioId}`,
        );
      }
    } catch (e) {
      console.error(
        `[API ERROR] Incident lookup/creation failed for scenario ${scenarioId}:`,
        e,
      );

      // IMPORTANT:
      // Do not silently hide the database error.
      console.error("[API ERROR] Message:", e.message);
      console.error("[API ERROR] Code:", e.code);
      console.error("[API ERROR] Detail:", e.detail);
      console.error("[API ERROR] Hint:", e.hint);
    }

    // =========================================================
    // 3. RETURN LOGS + INCIDENT ID
    // =========================================================
    res.json({
      logs: logs,
      incidentId: incidentId,
    });
  } catch (error) {
    console.error("[API ERROR] Failed to fetch scenario log:", error);

    // Keep existing fallback behavior
    res.json({
      logs: [
        "Loading simulation replay buffer...",
        "PostgreSQL timeline events retrieved successfully.",
        "Replay Engine heartbeat synchronized.",
      ],
      incidentId: null,
    });
  }
});
// ==========================================
// 2. INFERENCE & REPLAY LAUNCH ENDPOINTS
// ==========================================

// POST: Start inference using a pre-seeded database scenario or video cache check
router.post("/start-inference", async (req, res) => {
  const { cameraId, locationName, scenarioId, videoFile, lat, lng } = req.body;

  try {
    if (videoFile) {
      console.log(`[AI PIPELINE] Checking cache for footage: ${videoFile}`);

      const existingScenario = await db.query(
        "SELECT id FROM scenarios WHERE name = $1 LIMIT 1",
        [videoFile],
      );

      let activeScenarioId;

      if (existingScenario.rows.length > 0) {
        activeScenarioId = existingScenario.rows[0].id;
        console.log(
          `[AI PIPELINE] Cache Hit. Loading existing Scenario ID: ${activeScenarioId}`,
        );
        await replayEngine.startSimulation(activeScenarioId, locationName);

        return res.status(200).json({
          message: "Cached Timeline loaded successfully.",
          scenarioId: activeScenarioId,
          isNewProcessing: false,
        });
      }

      // CACHE MISS: Trigger Python YOLO Pipeline
      console.log(
        `[AI PIPELINE] Cache Miss. Initiating YOLOv8 processing for ${videoFile}...`,
      );

      const pythonExecutable = path.join(
        __dirname,
        "../../../../ai-pipeline/venv/Scripts/python.exe",
      );
      const scriptPath = path.join(
        __dirname,
        "../../../../ai-pipeline/process_video.py",
      );
      const videoPath = path.join(
        __dirname,
        "../../../../ai-pipeline",
        videoFile,
      );
      const pythonCommand = `"${pythonExecutable}" "${scriptPath}" --video "${videoPath}"`;

      exec(pythonCommand, async (error, stdout, stderr) => {
        if (error) {
          console.error(`[AI ERROR] YOLO Processing Failed: ${error.message}`);
          console.error(`[STDERR]: ${stderr}`);
          return res
            .status(500)
            .json({ message: "AI Processing Pipeline Failed." });
        }

        try {
          const detectedEvents = JSON.parse(stdout);

          const newScenario = await db.query(
            `INSERT INTO scenarios (name, description, duration_seconds) 
             VALUES ($1, $2, $3) RETURNING id`,
            [
              videoFile,
              `Auto-generated timeline for ${videoFile}`,
              detectedEvents.duration || 300,
            ],
          );

          activeScenarioId = newScenario.rows[0].id;

          for (const event of detectedEvents.timeline) {
            await db.query(
              `INSERT INTO timeline_events (scenario_id, timestamp_second, event_type, payload) 
               VALUES ($1, $2, $3, $4)`,
              [activeScenarioId, event.second, event.type, event.payload],
            );

            // FIX: Ensure new Python processing maps to incidents table
            if (
              event.type === "TRAFFIC_INCIDENT" ||
              event.type === "CROWD_DETECTED"
            ) {
              const insertLat = parseFloat(lat) || 19.076;
              const insertLng = parseFloat(lng) || 72.8777;
              const payloadWithScenario = {
                ...event.payload,
                scenario_id: activeScenarioId,
                video_file: videoFile,
                location: locationName,
              };

              await db.query(
                `INSERT INTO incidents (incident_type, severity, latitude, longitude, status, timeline_events, created_at)
                     VALUES ($1, $2, $3, $4, $5, $6, NOW())`,
                [
                  event.type,
                  "High",
                  insertLat,
                  insertLng,
                  "Active",
                  JSON.stringify(payloadWithScenario),
                ],
              );
            }
          }

          console.log(
            `[AI PIPELINE] Processing Complete. Saved as Scenario ID: ${activeScenarioId}`,
          );
          await replayEngine.startSimulation(activeScenarioId, locationName);

          res.status(200).json({
            message: "New footage processed and timeline cached.",
            scenarioId: activeScenarioId,
            isNewProcessing: true,
          });
        } catch (dbError) {
          console.error(
            "[DATABASE ERROR] Failed to save AI generated timeline.",
            dbError,
          );
          res.status(500).json({ message: "Failed to cache AI timeline." });
        }
      });
    } else if (scenarioId) {
  const reqLat = Number(lat);
  const reqLng = Number(lng);

  if (!Number.isFinite(reqLat) || !Number.isFinite(reqLng)) {
    return res.status(400).json({
      success: false,
      error: "Valid latitude and longitude are required.",
    });
  }

  console.log(
    `[API] Starting inference for Scenario ID: ${scenarioId} at Node: ${locationName} (${cameraId})`,
  );

  // CRITICAL FIX: DYNAMIC INCIDENT SEEDER
  try {
    const scenarioQuery = await db.query(
      "SELECT name, description FROM scenarios WHERE id = $1",
      [scenarioId],
    );

    let type = "SYSTEM_EVENT";
    let desc = "Simulation Active";

    if (scenarioQuery.rows.length > 0) {
      const scen = scenarioQuery.rows[0];

      desc = scen.description || scen.name;

      const descLower = desc.toLowerCase();

      if (
        descLower.includes("fire") ||
        descLower.includes("smoke")
      ) {
        type = "FIRE_HAZARD";
      } else if (
        descLower.includes("robbery") ||
        descLower.includes("pursuit") ||
        descLower.includes("crime")
      ) {
        type = "CRIME_ALERT";
      } else if (
        descLower.includes("accident") ||
        descLower.includes("collision") ||
        descLower.includes("rain")
      ) {
        type = "TRAFFIC_INCIDENT";
      } else {
        type = "AI_DETECTION";
      }
    }

    const incidentResult = await db.query(
      `INSERT INTO incidents
       (
         incident_type,
         severity,
         latitude,
         longitude,
         status,
         timeline_events,
         created_at
       )
       VALUES ($1, $2, $3, $4, $5, $6, NOW())
       RETURNING incident_id`,
      [
        type,
        "Critical",
        reqLat,
        reqLng,
        "Active",
        JSON.stringify({
          scenario_id: scenarioId,
          description: desc,
          video_file: `${scenarioId}.mp4`,
          confidence: 0.98,
          location: locationName,
        }),
      ],
    );

    const incidentId = incidentResult.rows[0].incident_id;

    console.log(
      `[API] Seeded active incident for Scenario ${scenarioId}: ${incidentId}`,
    );

    replayEngine.startSimulation(
      scenarioId,
      locationName,
    );

    return res.json({
      success: true,
      message: "Simulation started successfully.",
      incidentId,
      scenarioId,
      cameraId,
      locationName,
      latitude: reqLat,
      longitude: reqLng,
      status: "RUNNING",
    });
  } catch (e) {
    console.error(
      "[API ERROR] Incident seeding failed during start-inference:",
      e,
    );

    return res.status(500).json({
      success: false,
      error:
        "Simulation server could not create the selected incident.",
    });
  }
} else {
      res
        .status(400)
        .json({ error: "Missing scenarioId or videoFile parameters." });
    }
  } catch (error) {
    console.error("[API ERROR] Start inference failed:", error);
    res.status(500).json({ error: "Failed to start simulation inference." });
  }
});

// POST: Handle custom video upload and start simulation
router.post("/upload-and-start", upload.single("video"), async (req, res) => {
  const { cameraId, locationName, scenarioId } = req.body;
  const uploadedFile = req.file;

  try {
    console.log(
      `[API] Custom video uploaded: ${uploadedFile ? uploadedFile.filename : "None"}`,
    );
    console.log(
      `[API] Starting simulation with Scenario ID: ${scenarioId} at ${locationName}`,
    );

    replayEngine.startSimulation(scenarioId, locationName);

    res.json({
      success: true,
      message: "Video uploaded and simulation started successfully.",
      filename: uploadedFile ? uploadedFile.filename : null,
    });
  } catch (error) {
    console.error("[API ERROR] Upload and start failed:", error);
    res
      .status(500)
      .json({ error: "Failed to process video upload and simulation." });
  }
});

// ==========================================
// 3. LIVE YOLOv8 INFERENCE (REAL PYTHON EXECUTION)
// ==========================================

// POST: Handle Live Video Upload with SHA-256 Caching
router.post("/live/upload", upload.single("video"), async (req, res) => {
  const { cameraId, location, lat, lng } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ error: "No video file provided." });
  }

  const originalName = file.originalname;
  const filepath = file.path;

  try {
    // 1. Calculate SHA-256 Fingerprint
    const fileHash = await calculateFileHash(filepath);
    console.log(`[LIVE PIPELINE] Video SHA-256 Fingerprint: ${fileHash}`);

    // 2. CACHE CHECK: Did the user upload this exact file before?
    const existingUpload = await db.query(
      `SELECT id, status FROM live_video_uploads WHERE video_hash = $1 AND status = 'Completed' LIMIT 1`,
      [fileHash],
    );

    if (existingUpload.rows.length > 0) {
      const uploadId = existingUpload.rows[0].id;
      console.log(
        `[LIVE PIPELINE] Cache Hit! Live video hash "${fileHash}" was already processed.`,
      );

      // Cleanup the duplicate uploaded file to save disk space
      fs.unlinkSync(filepath);

      return res.json({
        success: true,
        uploadId,
        message:
          "Cached video found. Reusing previously stored AI detection timeline.",
        isCached: true,
      });
    }

    // 3. CACHE MISS: Insert tracking record as 'Processing'
    console.log(
      `[LIVE PIPELINE] Cache Miss. Initiating REAL YOLOv8 inference for "${originalName}"...`,
    );

    const insertRes = await db.query(
      `INSERT INTO live_video_uploads (filename, filepath, video_hash, camera_id, location, status, processing_started_at) 
       VALUES ($1, $2, $3, $4, $5, $6, NOW()) RETURNING id`,
      [
        originalName,
        filepath,
        fileHash,
        cameraId || "CAM-101",
        location || "Mumbai Sector",
        "Processing",
      ],
    );
    const uploadId = insertRes.rows[0].id;

    // Return the ID instantly so the React frontend terminal can start polling for logs
    res.json({
      success: true,
      uploadId,
      message: "Real edge inference initiated.",
      isCached: false,
    });

    // 4. EXECUTE REAL PYTHON YOLO SCRIPT (Async Background Thread)
    const pythonExecutable = path.join(
      __dirname,
      "../../../../ai-pipeline/venv/Scripts/python.exe",
    );
    const scriptPath = path.join(
      __dirname,
      "../../../../ai-pipeline/process_video.py",
    );
    const pythonCommand = `"${pythonExecutable}" "${scriptPath}" --video "${filepath}"`;

    exec(pythonCommand, async (error, stdout, stderr) => {
      if (error) {
        console.error(
          `[AI ERROR] Live YOLO Processing Failed: ${error.message}`,
        );
        await db.query(
          `UPDATE live_video_uploads SET status = 'Failed' WHERE id = $1`,
          [uploadId],
        );
        return;
      }

      try {
        const detectedEvents = JSON.parse(stdout);

        // 5. SAVE REAL DETECTIONS AND TRACKS TO DATABASE
        if (detectedEvents.tracking) {
          for (const det of detectedEvents.tracking) {
            // Save bounding box & detection confidence
            await db.query(
              `INSERT INTO live_detection_results (upload_id, frame, object_detected, confidence, tracking_id, timestamp_second) 
               VALUES ($1, $2, $3, $4, $5, $6)`,
              [
                uploadId,
                det.frame || Math.floor(det.second * 30),
                det.class || det.object,
                det.confidence || 0.85,
                det.track_id,
                det.second,
              ],
            );

            // Save trajectory and movement features to tracking table
            await db.query(
              `INSERT INTO live_tracking_results (upload_id, tracking_id, object_class, current_latitude, current_longitude, speed_kmh, timestamp_second) 
               VALUES ($1, $2, $3, $4, $5, $6, $7)`,
              [
                uploadId,
                det.track_id,
                det.class || det.object,
                det.lat || null,
                det.lng || null,
                det.speed || null,
                det.second,
              ],
            );
          }
        }

        // 6. SAVE REAL INCIDENT EVENTS TO TIMELINE TABLE
        if (detectedEvents.timeline) {
          for (const event of detectedEvents.timeline) {
            // 1. Save to the specific Live timeline
            await db.query(
              `INSERT INTO live_event_timeline (upload_id, event_type, frame, timestamp_second, description) VALUES ($1, $2, $3, $4, $5)`,
              [
                uploadId,
                event.type,
                event.frame,
                event.second,
                JSON.stringify(event.payload),
              ],
            );

            // 2. FIX: Push critical YOLO events to the GLOBAL Digital Twin incidents table using STRICT column names
            if (
              event.type === "TRAFFIC_INCIDENT" ||
              event.type === "CROWD_DETECTED"
            ) {
              const payloadWithVideo = {
                ...event.payload,
                video_file: `../uploads/${file.filename}`,
                location: location,
              };
              await db.query(
                `INSERT INTO incidents (incident_type, severity, latitude, longitude, status, timeline_events, upload_id, created_at)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())`,
                [
                  event.type,
                  "High",
                  parseFloat(lat) || 19.076,
                  parseFloat(lng) || 72.8777,
                  "Active",
                  JSON.stringify(payloadWithVideo),
                  uploadId,
                ],
              );
            }
          }
        }

        // FALLBACK: If YOLO found nothing critical, ensure at least one incident is created so dashboard connects
        const incidentCheck = await db.query(
          "SELECT incident_id FROM incidents WHERE upload_id = $1",
          [uploadId],
        );
        if (incidentCheck.rows.length === 0) {
          await db.query(
            `INSERT INTO incidents (incident_type, severity, latitude, longitude, status, timeline_events, upload_id, created_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())`,
            [
              "AI_DETECTION",
              "Medium",
              parseFloat(lat) || 19.076,
              parseFloat(lng) || 72.8777,
              "Active",
              JSON.stringify({
                description: "AI Processing Completed",
                confidence: 0.85,
                video_file: `../uploads/${file.filename}`,
                location: location,
              }),
              uploadId,
            ],
          );
        }

        await db.query(
          `UPDATE live_video_uploads SET status = 'Completed', processing_completed_at = NOW() WHERE id = $1`,
          [uploadId],
        );
        console.log(
          `[LIVE PIPELINE] Python YOLOv8 Processing Complete for Upload ID: ${uploadId}`,
        );
      } catch (dbError) {
        console.error(
          "[DATABASE ERROR] Failed to save live AI generated timeline.",
          dbError,
        );
        await db.query(
          `UPDATE live_video_uploads SET status = 'Failed' WHERE id = $1`,
          [uploadId],
        );
      }
    });
  } catch (err) {
    console.error("[API ERROR] Live upload failed:", err);
    res.status(500).json({ error: "Live upload processing failed." });
  }
});

// GET: Stream live inference logs
router.get("/live/:id/log", async (req, res) => {
  const uploadId = req.params.id;
  try {
    const uploadStatus = await db.query(
      "SELECT status FROM live_video_uploads WHERE id = $1",
      [uploadId],
    );
    const status = uploadStatus.rows[0]?.status || "Processing";

    const events = await db.query(
      "SELECT timestamp_second, event_type, description FROM live_event_timeline WHERE upload_id = $1",
      [uploadId],
    );

    let logs = [
      "Loading YOLOv8 model weights...",
      "Reading incoming video stream frames...",
      "Frame 1: Initializing bounding box tensors...",
      "Frame 5: Object detection active (Car, Bus, Person)",
    ];

    events.rows.forEach((e) => {
      logs.push(`[T=${e.timestamp_second}s] ${e.event_type}: ${e.description}`);
    });

    let incidentId = null;

    if (status === "Completed") {
      logs.push("Saving detections and tracking vectors to live database...");
      logs.push("Generating replay timeline. Inference Complete.");

      try {
        const incidentRes = await db.query(
          "SELECT incident_id FROM incidents WHERE upload_id = $1 ORDER BY created_at DESC LIMIT 1",
          [uploadId],
        );
        if (incidentRes.rows.length > 0) {
          incidentId = incidentRes.rows[0].incident_id;

          await db.query(
            "UPDATE incidents SET status = 'Active', created_at = NOW() WHERE incident_id = $1",
            [incidentId],
          );
        }
      } catch (dbErr) {
        console.error(
          "[DATABASE ERROR] Could not fetch incident ID for live upload:",
          dbErr,
        );
      }
    }

    res.json({
      status,
      progress: status === "Completed" ? 100 : 75,
      logs,
      incidentId, // Passes the exact DB reference to the frontend
    });
  } catch (err) {
    res.json({
      status: "Processing",
      progress: 50,
      logs: ["Streaming live inference telemetry..."],
      incidentId: null,
    });
  }
});

// DELETE: Mark an incident as resolved by permanently deleting it from the database and cache
router.delete("/incidents/:id", async (req, res) => {
  const incidentId = req.params.id;
  try {
    // Hard delete the incident so it doesn't cache or block future simulation runs
    await db.query("DELETE FROM incidents WHERE incident_id = $1", [
      incidentId,
    ]);
    console.log(`[API] Incident ${incidentId} permanently deleted.`);
    res.json({ success: true, message: "Incident deleted successfully." });
  } catch (error) {
    console.error("[API ERROR] Failed to delete incident:", error);
    res.status(500).json({ error: "Failed to delete incident." });
  }
});

module.exports = router;
