export const DIRECTIONS = ["Northbound", "Southbound", "Eastbound", "Westbound", "Intersection"];

export const MUMBAI_ROADS = [
  "S.V. Road", "New Link Road", "LBS Marg", "Western Express Hwy",
  "Eastern Express Hwy", "Veer Savarkar Marg", "Dr. Annie Besant Road",
  "Senapati Bapat Marg", "Tulsi Pipe Road", "M.G. Road", "Dr. D.N. Road",
  "P. D'Mello Road", "C.S.T. Road", "Juhu Tara Road", "Pali Hill Road",
];

export const getCleanAreaName = (landmarkName) => {
  if (!landmarkName) return "Mumbai";
  if (landmarkName.includes(",")) return landmarkName.split(",").pop().trim();

  return landmarkName
    .replace(/\b(Police Station|Hospital|Fire Dept|Station)\b/gi, "")
    .replace(/\(.*?\)/g, "")
    .replace(/[0-9]/g, "")
    .trim() || "Mumbai";
};

export const getNearestLandmark = (landmarks, latitude, longitude) => {
  if (!landmarks.length) return "Mumbai";

  return landmarks.reduce((nearest, landmark) => {
    const landmarkLat = Number(landmark.latitude);
    const landmarkLng = Number(landmark.longitude);
    const distance = (landmarkLat - latitude) ** 2 + (landmarkLng - longitude) ** 2;
    return distance < nearest.distance ? { name: landmark.name, distance } : nearest;
  }, { name: "Mumbai", distance: Infinity }).name;
};

export const buildCameras = ({ entities, feedSource, availableVideos, externalFeeds }) => {
  const landmarks = entities.filter(({ entity_type }) =>
    entity_type === "hospital" || entity_type === "police_station",
  );
  const cameraEntities = entities.filter(({ entity_type }) => entity_type === "cctv");

  return cameraEntities.map((entity, index) => {
    const latitude = Number(entity.latitude) || 19.076;
    const longitude = Number(entity.longitude) || 72.8777;
    const isOffline = index % 7 === 0;
    const status = isOffline ? "Offline" : "Online";
    const isVideo = feedSource === "loop";
    const streamUrl = isOffline
      ? null
      : isVideo
        ? availableVideos[index % availableVideos.length]
        : externalFeeds[index % externalFeeds.length]?.image ?? null;
    const locationName = getCleanAreaName(getNearestLandmark(landmarks, latitude, longitude));
    const roadName = MUMBAI_ROADS[index % MUMBAI_ROADS.length];

    return {
      camera_id: entity.entity_id,
      camera_code: entity.name || `CAM-MUM-${String(index + 1).padStart(3, "0")}`,
      name: `${roadName}, ${locationName}`,
      location_name: locationName,
      latitude,
      longitude,
      road_name: roadName,
      direction: DIRECTIONS[index % DIRECTIONS.length],
      status,
      source: isVideo ? "Local Edge Storage" : "External Open API",
      stream_type: isVideo ? "Live video stream" : "API snapshot",
      stream_url: streamUrl,
      isVideo,
    };
  });
};
