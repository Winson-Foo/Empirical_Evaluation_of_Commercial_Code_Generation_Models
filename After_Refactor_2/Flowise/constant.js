// UI constants
export const SPACING_GRID = 3;
export const WIDTH_DRAWER = 260;
export const WIDTH_APP_DRAWER = 320;
export const MAX_SCROLL = 100000;

//Network constants
export const BASE_URL = process.env.NODE_ENV === 'production' ? window.location.origin : window.location.origin.replace(':8080', ':3000');
export const UI_BASE_URL = window.location.origin;

