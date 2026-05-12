export interface ZoneData {
  id: string;
  name: string;
  centroid: [number, number]; // [lng, lat]
  polygon: [number, number][]; // closed GeoJSON ring
  nearVenues: string[];
  baseLoad: number; // 0–1 typical traffic load
}

// Approximate neighborhood polygon boundaries for key LA districts.
// Coordinates are [longitude, latitude]. Each ring is closed (first === last).
export const LA_ZONES: ZoneData[] = [
  {
    id: 'inglewood',
    name: 'Inglewood',
    centroid: [-118.353, 33.962],
    polygon: [
      [-118.388, 33.918], [-118.312, 33.916], [-118.304, 33.941],
      [-118.296, 33.968], [-118.308, 33.993], [-118.335, 33.997],
      [-118.362, 33.992], [-118.381, 33.972], [-118.388, 33.950],
      [-118.388, 33.918],
    ],
    nearVenues: ['sofi', 'intuit-dome'],
    baseLoad: 0.62,
  },
  {
    id: 'hawthorne',
    name: 'Hawthorne / Lawndale',
    centroid: [-118.352, 33.900],
    polygon: [
      [-118.389, 33.882], [-118.318, 33.879], [-118.308, 33.910],
      [-118.312, 33.922], [-118.370, 33.924], [-118.389, 33.918],
      [-118.389, 33.882],
    ],
    nearVenues: ['sofi', 'intuit-dome'],
    baseLoad: 0.45,
  },
  {
    id: 'crenshaw',
    name: 'Crenshaw / Hyde Park',
    centroid: [-118.332, 34.006],
    polygon: [
      [-118.358, 33.990], [-118.305, 33.988], [-118.296, 34.005],
      [-118.296, 34.022], [-118.315, 34.030], [-118.345, 34.028],
      [-118.362, 34.018], [-118.362, 33.998], [-118.358, 33.990],
    ],
    nearVenues: ['sofi', 'la-coliseum', 'bmo-stadium'],
    baseLoad: 0.52,
  },
  {
    id: 'south-la',
    name: 'South LA / Watts',
    centroid: [-118.248, 33.954],
    polygon: [
      [-118.290, 33.920], [-118.210, 33.918], [-118.205, 33.955],
      [-118.210, 33.980], [-118.250, 33.985], [-118.288, 33.978],
      [-118.295, 33.958], [-118.290, 33.920],
    ],
    nearVenues: ['la-coliseum', 'dignity-health'],
    baseLoad: 0.40,
  },
  {
    id: 'exposition-park',
    name: 'Exposition Park / USC',
    centroid: [-118.280, 34.018],
    polygon: [
      [-118.305, 33.998], [-118.252, 33.997], [-118.248, 34.018],
      [-118.250, 34.036], [-118.268, 34.042], [-118.295, 34.038],
      [-118.308, 34.025], [-118.305, 33.998],
    ],
    nearVenues: ['la-coliseum', 'bmo-stadium'],
    baseLoad: 0.58,
  },
  {
    id: 'downtown',
    name: 'Downtown LA',
    centroid: [-118.250, 34.050],
    polygon: [
      [-118.275, 34.028], [-118.226, 34.030], [-118.222, 34.052],
      [-118.224, 34.072], [-118.240, 34.078], [-118.262, 34.075],
      [-118.272, 34.062], [-118.272, 34.040], [-118.275, 34.028],
    ],
    nearVenues: ['crypto-arena'],
    baseLoad: 0.72,
  },
  {
    id: 'koreatown',
    name: 'Koreatown / Mid-Wilshire',
    centroid: [-118.302, 34.062],
    polygon: [
      [-118.335, 34.040], [-118.270, 34.040], [-118.268, 34.058],
      [-118.272, 34.075], [-118.300, 34.078], [-118.330, 34.074],
      [-118.338, 34.060], [-118.335, 34.040],
    ],
    nearVenues: ['crypto-arena', 'la-coliseum'],
    baseLoad: 0.60,
  },
  {
    id: 'hollywood',
    name: 'Hollywood',
    centroid: [-118.315, 34.090],
    polygon: [
      [-118.358, 34.072], [-118.268, 34.072], [-118.265, 34.092],
      [-118.270, 34.108], [-118.295, 34.112], [-118.325, 34.110],
      [-118.352, 34.100], [-118.360, 34.085], [-118.358, 34.072],
    ],
    nearVenues: ['crypto-arena'],
    baseLoad: 0.55,
  },
  {
    id: 'west-hollywood',
    name: 'West Hollywood / Fairfax',
    centroid: [-118.368, 34.088],
    polygon: [
      [-118.398, 34.072], [-118.338, 34.072], [-118.335, 34.088],
      [-118.340, 34.102], [-118.360, 34.106], [-118.390, 34.100],
      [-118.400, 34.088], [-118.398, 34.072],
    ],
    nearVenues: ['crypto-arena', 'sofi'],
    baseLoad: 0.50,
  },
  {
    id: 'westwood',
    name: 'Westwood / UCLA',
    centroid: [-118.445, 34.068],
    polygon: [
      [-118.468, 34.048], [-118.428, 34.048], [-118.425, 34.062],
      [-118.428, 34.082], [-118.440, 34.088], [-118.458, 34.085],
      [-118.468, 34.072], [-118.468, 34.048],
    ],
    nearVenues: ['pauley', 'ucla-olympic', 'sepulveda-basin'],
    baseLoad: 0.55,
  },
  {
    id: 'culver-city',
    name: 'Culver City / Mar Vista',
    centroid: [-118.400, 34.018],
    polygon: [
      [-118.432, 33.998], [-118.368, 33.998], [-118.365, 34.018],
      [-118.368, 34.038], [-118.390, 34.042], [-118.418, 34.038],
      [-118.432, 34.022], [-118.432, 33.998],
    ],
    nearVenues: ['sofi', 'pauley'],
    baseLoad: 0.48,
  },
  {
    id: 'brentwood',
    name: 'Brentwood / West LA',
    centroid: [-118.475, 34.042],
    polygon: [
      [-118.500, 34.022], [-118.445, 34.022], [-118.442, 34.040],
      [-118.445, 34.060], [-118.462, 34.065], [-118.488, 34.060],
      [-118.500, 34.048], [-118.500, 34.022],
    ],
    nearVenues: ['pauley', 'ucla-olympic'],
    baseLoad: 0.38,
  },
  {
    id: 'pasadena',
    name: 'Pasadena / Altadena',
    centroid: [-118.158, 34.152],
    polygon: [
      [-118.205, 34.128], [-118.112, 34.128], [-118.108, 34.158],
      [-118.115, 34.182], [-118.148, 34.192], [-118.188, 34.185],
      [-118.205, 34.165], [-118.205, 34.128],
    ],
    nearVenues: ['rose-bowl'],
    baseLoad: 0.40,
  },
  {
    id: 'van-nuys',
    name: 'Van Nuys / North Hollywood',
    centroid: [-118.455, 34.188],
    polygon: [
      [-118.498, 34.168], [-118.412, 34.168], [-118.410, 34.192],
      [-118.415, 34.212], [-118.445, 34.215], [-118.482, 34.210],
      [-118.498, 34.195], [-118.498, 34.168],
    ],
    nearVenues: ['sepulveda-basin'],
    baseLoad: 0.42,
  },
  {
    id: 'encino',
    name: 'Encino / Sherman Oaks',
    centroid: [-118.490, 34.158],
    polygon: [
      [-118.528, 34.138], [-118.452, 34.138], [-118.448, 34.158],
      [-118.452, 34.175], [-118.472, 34.180], [-118.510, 34.178],
      [-118.528, 34.165], [-118.528, 34.138],
    ],
    nearVenues: ['sepulveda-basin'],
    baseLoad: 0.35,
  },
  {
    id: 'long-beach',
    name: 'Long Beach',
    centroid: [-118.188, 33.786],
    polygon: [
      [-118.225, 33.762], [-118.148, 33.758], [-118.142, 33.782],
      [-118.148, 33.808], [-118.178, 33.820], [-118.212, 33.815],
      [-118.228, 33.800], [-118.225, 33.762],
    ],
    nearVenues: ['long-beach-arena', 'el-dorado'],
    baseLoad: 0.38,
  },
  {
    id: 'carson',
    name: 'Carson / Torrance',
    centroid: [-118.268, 33.862],
    polygon: [
      [-118.308, 33.840], [-118.228, 33.838], [-118.222, 33.858],
      [-118.225, 33.882], [-118.252, 33.892], [-118.292, 33.888],
      [-118.312, 33.872], [-118.308, 33.840],
    ],
    nearVenues: ['dignity-health'],
    baseLoad: 0.35,
  },
  {
    id: 'airport',
    name: 'LAX / Airport District',
    centroid: [-118.388, 33.948],
    polygon: [
      [-118.420, 33.928], [-118.358, 33.928], [-118.352, 33.942],
      [-118.355, 33.960], [-118.372, 33.968], [-118.405, 33.965],
      [-118.422, 33.952], [-118.420, 33.928],
    ],
    nearVenues: ['sofi', 'intuit-dome'],
    baseLoad: 0.68,
  },
];
