/**
 * Modern Google Maps Routes API (V2) Integration
 * Replaces the legacy google.maps.DirectionsService
 */
export async function calculateModernRoutes(
    originAddress,
    destinationAddress,
    apiKey,
    options = {}
) {
    if (!apiKey) {
        throw new Error("Missing Google Maps API Key.");
    }

    const {
        departureTime,
        trafficModel = "BEST_GUESS",
        regionCode = "IN",
        languageCode = "en-IN",
        simple = false,
        _retried = false,
    } = options;

    // Google only returns a genuinely *predicted* (as opposed to purely live)
    // traffic-aware duration when routingPreference is TRAFFIC_AWARE_OPTIMAL
    // and a future departureTime + trafficModel are supplied.
    //
    // Plain TRAFFIC_AWARE reflects current/live traffic.
    //
    // Google does not provide a public API for requesting a genuine historical
    // traffic snapshot for an arbitrary past date/time.
    const isPredictive = !!departureTime;

    const requestBody = {
        origin: {
            address: originAddress,
        },

        destination: {
            address: destinationAddress,
        },

        travelMode: "DRIVE",

        // Bias address resolution to India so short names like "Dadar" resolve correctly
        regionCode,
        languageCode,

        routingPreference: isPredictive
            ? "TRAFFIC_AWARE_OPTIMAL"
            : "TRAFFIC_AWARE",

        computeAlternativeRoutes: !simple,

        // Ask Google Routes API for traffic classification on the route
        // polyline.
        ...(simple ? {} : { extraComputations: ["TRAFFIC_ON_POLYLINE"] }),

        ...(isPredictive
            ? {
                  departureTime,
                  trafficModel,
              }
            : {}),
    };

    try {
        const response = await fetch(
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",

                    "X-Goog-Api-Key": apiKey,

                    /*
                     * Request only the fields required by the
                     * LiveTraffic UI and Traffic Agent.
                     *
                     * duration:
                     *   Traffic-aware / predicted travel time.
                     *
                     * staticDuration:
                     *   Traffic-free baseline duration.
                     *
                     * speedReadingIntervals:
                     *   Google's traffic classification along
                     *   the route.
                     */
                    "X-Goog-FieldMask":
                        "routes.routeLabels," +
                        "routes.distanceMeters," +
                        "routes.duration," +
                        "routes.staticDuration," +
                        "routes.polyline.encodedPolyline," +
                        "routes.travelAdvisory.speedReadingIntervals",
                },

                body: JSON.stringify(requestBody),
            }
        );

        if (!response.ok) {
            const errorData = await response
                .json()
                .catch(() => ({}));

            console.error(
                "Google Routes API Technical Error:",
                errorData
            );

            throw new Error(
                "Route service unavailable. Check Google Maps Routes API configuration."
            );
        }

        const data = await response.json();

        if (!data.routes || data.routes.length === 0) {
            console.warn("Routes API returned no routes", {
                originAddress,
                destinationAddress,
                simple,
                response: data,
            });

            // Retry once with a minimal request (no alternatives / polyline traffic)
            if (!_retried) {
                return calculateModernRoutes(originAddress, destinationAddress, apiKey, {
                    ...options,
                    simple: true,
                    _retried: true,
                });
            }
            return [];
        }

        /*
         * Normalize Google's response for LiveTraffic.jsx
         * and TrafficAgent backend.
         */
        return data.routes.map((route, index) => {
            const distanceKm = (
                route.distanceMeters / 1000
            ).toFixed(1);

            /*
             * Google duration format:
             *
             * "1200s"
             */
            const durationSeconds = parseInt(
                route.duration?.replace("s", "") || 0,
                10
            );

            const durationMins = Math.ceil(
                durationSeconds / 60
            );

            /*
             * Google's traffic-free duration for the same route.
             *
             * Comparing duration with staticDuration allows
             * us to calculate the traffic delay without
             * fabricating a value.
             */
            const staticDurationSeconds =
                parseInt(
                    route.staticDuration?.replace("s", "") || 0,
                    10
                ) || durationSeconds;

            const delayMinutes = Math.max(
                0,
                Math.round(
                    (durationSeconds -
                        staticDurationSeconds) /
                        60
                )
            );

            /*
             * Average speed derived from Google's route
             * distance and traffic-aware duration.
             */
            const averageSpeedKmh =
                durationSeconds > 0
                    ? Math.round(
                          (route.distanceMeters / 1000) /
                              (durationSeconds / 3600)
                      )
                    : null;

            /*
             * Clean Google's route labels.
             *
             * Example:
             *
             * DEFAULT_ROUTE
             *
             * becomes:
             *
             * Default Route
             */
            const label =
                route.routeLabels &&
                route.routeLabels.length > 0
                    ? route.routeLabels[0]
                          .replace(/_/g, " ")
                          .toLowerCase()
                          .replace(
                              /\b\w/g,
                              (letter) =>
                                  letter.toUpperCase()
                          )
                    : "Alternative Route";

            return {
                id: index,

                name: `Route ${String.fromCharCode(
                    65 + index
                )}`,

                summary: label,

                distance: `${distanceKm} km`,

                duration: `${durationMins} mins`,

                durationSeconds,

                staticDurationSeconds,

                delayMinutes,

                averageSpeedKmh,

                /*
                 * true when departureTime was supplied.
                 *
                 * This is used by LiveTraffic.jsx to identify
                 * Google predicted traffic.
                 */
                isPredicted: isPredictive,

                /*
                 * Google's encoded route polyline.
                 */
                polyline:
                    route.polyline?.encodedPolyline,

                /*
                 * Google traffic classification along
                 * the route.
                 *
                 * Typical values include:
                 *
                 * NORMAL
                 * SLOW
                 * TRAFFIC_JAM
                 */
                speedReadingIntervals:
                    route.travelAdvisory
                        ?.speedReadingIntervals || [],

                /*
                 * Preserve the complete Google route response
                 * for advanced processing/debugging.
                 */
                rawRoute: route,
            };
        });
    } catch (error) {
        console.error(
            "Routing Service Failure:",
            error
        );

        throw new Error(
            "Route service unavailable. Check Google Maps Routes API configuration."
        );
    }
}