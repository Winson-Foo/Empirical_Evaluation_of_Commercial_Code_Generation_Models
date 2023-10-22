// config.js
export const SPACING_GRID = 3
export const WIDTH_DRAWER = 260
export const WIDTH_APP_DRAWER = 320
export const MAX_SCROLL = 100000
export const BASE_URL = process.env.NODE_ENV === 'production' ? window.location.origin : window.location.origin.replace(':8080', ':3000')
export const UI_BASE_URL = window.location.origin

// component.js
import { SPACING_GRID, WIDTH_DRAWER, WIDTH_APP_DRAWER, MAX_SCROLL, BASE_URL, UI_BASE_URL } from './config'

// use the constants in your code
console.log(SPACING_GRID) // output: 3
console.log(WIDTH_DRAWER) // output: 260
console.log(BASE_URL) // output: window.location.origin or window.location.origin.replace(':8080', ':3000') depending on the environment