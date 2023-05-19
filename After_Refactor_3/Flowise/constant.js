export const constants = {
  gridSpacing: 3,
  drawerWidth: 260,
  appDrawerWidth: 320,
  maxScroll: 100000,
  baseURL: process.env.NODE_ENV === 'production' ?
    window.location.origin :
    window.location.origin.replace(':8080', ':3000'),
  uiBaseURL: window.location.origin
};