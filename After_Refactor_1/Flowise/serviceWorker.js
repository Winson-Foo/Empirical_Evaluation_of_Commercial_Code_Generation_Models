const LOCALHOST_NAMES = ['localhost', '[::1]', /^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/];
const UPDATE_MESSAGE = 'New content is available and will be used when all tabs for this page are closed. See https://bit.ly/CRA-PWA.';
const CACHED_MESSAGE = 'Content is cached for offline use.';
const ERROR_MESSAGE = 'Error during service worker registration:';
const NO_INTERNET_MESSAGE = 'No internet connection found. App is running in offline mode.';
const SERVICE_WORKER_HEADER = 'Service-Worker';
const JAVASCRIPT_CONTENT_TYPE = 'javascript';
const SERVICE_WORKER_JS = 'service-worker.js';

function isLocalhost() {
  return LOCALHOST_NAMES.includes(window.location.hostname);
}

function handleUpdateFound(installingWorker, config) {
  installingWorker.onstatechange = () => {
    if (installingWorker.state === 'installed') {
      if (navigator.serviceWorker.controller) {
        console.info(UPDATE_MESSAGE);

        if (config && config.onUpdate) {
          config.onUpdate(installingWorker);
        }
      } else {
        console.info(CACHED_MESSAGE);

        if (config && config.onSuccess) {
          config.onSuccess(installingWorker);
        }
      }
    }
  }
}

function registerValidSW(swUrl, config) {
  navigator.serviceWorker.register(swUrl)
    .then((registration) => {
      registration.onupdatefound = () => handleUpdateFound(registration.installing, config);
    })
    .catch((error) => {
      console.error(ERROR_MESSAGE, error);
    });
}

function unregisterServiceWorker() {
  navigator.serviceWorker.ready
    .then(registration => registration.unregister())
    .catch(error => console.error(error.message));
}

function checkServiceWorker(swUrl, config) {
  fetch(swUrl, { headers: { [SERVICE_WORKER_HEADER]: 'script' } })
    .then((response) => {
      const contentType = response.headers.get('content-type');

      if (response.status === 404 || (contentType && contentType.indexOf(JAVASCRIPT_CONTENT_TYPE) === -1)) {
        unregisterServiceWorker(); // Unregister the service worker and reload the page.
        window.location.reload();
      } else {
        registerValidSW(swUrl, config); // Proceed as normal.
      }
    })
    .catch(() => console.info(NO_INTERNET_MESSAGE));
}

export function register(config) {
  if (process.env.NODE_ENV === 'production' && 'serviceWorker' in navigator) {
    const PUBLIC_URL = new URL(process.env.PUBLIC_URL, window.location.href);

    if (PUBLIC_URL.origin !== window.location.origin) {
      return; // Our service worker won't work if PUBLIC_URL is on a different origin.
    }

    window.addEventListener('load', () => {
      const swUrl = `${process.env.PUBLIC_URL}/${SERVICE_WORKER_JS}`;

      if (isLocalhost()) {
        checkServiceWorker(swUrl, config);
        navigator.serviceWorker.ready.then(() => {
          console.info(`This web app is being served cache-first by a service worker. To learn more, visit https://bit.ly/CRA-PWA`);
        });
      } else {
        registerValidSW(swUrl, config); // Just register the service worker.
      }
    });
  }
}

export function unregister() {
  if ('serviceWorker' in navigator) {
    unregisterServiceWorker();
  }
}

