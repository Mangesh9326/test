export const AI_TASKS = [
  "Traffic flow analysis",
  "Vehicle density analysis",
  "Queue detection",
  "Lane occupancy analysis",
  "Incident watch",
  "Pedestrian activity",
  "Congestion watch",
  "Continuous monitoring"
];

export function getDeterministicTask(cameraId, tick) {
  let hash = 0;
  for (let i = 0; i < cameraId.length; i++) {
    hash = (hash << 5) - hash + cameraId.charCodeAt(i);
    hash |= 0;
  }
  const index = Math.abs(hash + tick) % AI_TASKS.length;
  return AI_TASKS[index];
}

export function buildCameras({ entities, feedSource, availableVideos, externalFeeds }) {
  const cctvEntities = entities.filter(e => e.entity_type === 'cctv');
  return cctvEntities.map((e, index) => {
    // Deterministic pseudo-random status assignment based on index to ensure stable simulation
    const isOffline = index % 11 === 0; 
    const status = isOffline ? "Offline" : "Online";

    let stream_url = null;
    let isVideo = false;

    if (!isOffline) {
      if (feedSource === 'loop') {
        stream_url = availableVideos[index % availableVideos.length];
        isVideo = true;
      } else {
         stream_url = externalFeeds[index % externalFeeds.length]?.image_metadata?.image_url 
                   || externalFeeds[index % externalFeeds.length]?.image;
         isVideo = false;
      }
    }

    return {
      camera_id: e.entity_id,
      camera_code: e.name || `CAM-MUM-${String(index + 1).padStart(3, '0')}`,
      name: `${e.metadata?.road_name || 'Mumbai Road'} Traffic Node`,
      location_name: e.metadata?.road_name || 'Mumbai',
      road_name: e.metadata?.road_name || 'Mumbai',
      latitude: parseFloat(e.latitude) || 19.0760,
      longitude: parseFloat(e.longitude) || 72.8777,
      direction: e.metadata?.direction || ['Northbound', 'Southbound', 'Eastbound', 'Westbound'][index % 4],
      status,
      source: 'Municipal Traffic Control',
      stream_type: isVideo ? 'HLS/RTSP' : 'Live Snapshot',
      stream_url,
      isVideo
    };
  });
}