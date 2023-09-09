const LOCALHOST_NAMES = ['localhost', '[::1]'];
const IPV4_PATTERN = /^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/;

const MESSAGES = {
  OFFLINE: 'No internet connection found. App is running in offline mode.',
  CONTENT_CACHED: 'Content is cached for offline use.',
  NEW_CONTENT_AVAILABLE:
    'New content is available and will be used when all tabs for this page are closed. See https://bit.ly/CRA-PWA.',
  REGISTRATION_ERROR: 'Error during service worker registration:',
};

function registerValidSW(swUrl, { onUpdate, onSuccess }) {
  navigator.serviceWorker.register(swUrl).then(registration => {
    registration.onupdatefound = () => {
      const installingWorker = registration.installing;

      if (installingWorker === null) {
        return;
      }

      installingWorker.onstatechange = () => {
        if (installingWorker.state === 'installed') {
          if (navigator.serviceWorker.controller) {
            console.info(MESSAGES.NEW_CONTENT_AVAILABLE);

            if (onUpdate) {
              onUpdate(registration);
            }
          } else {
            console.info(MESSAGES.CONTENT_CACHED);

            if (onSuccess) {
              onSuccess(registration);
            }
          }
        }
      };
    };
  }).catch(error => {
    console.error(`${MESSAGES.REGISTRATION_ERROR} ${error}`);
  });
}

function checkValidServiceWorker(url, config) {
  fetch(url, { headers: { 'Service-Worker': 'script' } })
    .then(response => {
      const contentType = response.headers.get('content-type');

      if (
        response.status === 404 ||
        (contentType != null && contentType.indexOf('javascript') === -1)
      ) {
        navigator.serviceWorker.ready
          .then(registration => registration.unregister())
          .then(() => window.location.reload());
      } else {
        registerValidSW(url, config);
      }
    })
    .catch(() => {
      console.info(MESSAGES.OFFLINE);
    });
}

export const register = config => {
  if (
    process.env.NODE_ENV === 'production' &&
    'serviceWorker' in navigator
  ) {
    const url = `${process.env.PUBLIC_URL}/service-worker.js`;
    const { origin } = new URL(process.env.PUBLIC_URL, window.location.href);

    if (origin !== window.location.origin) {
      return;
    }

    window.addEventListener('load', () => {
      if (LOCALHOST_NAMES.includes(window.location.hostname)) {
        checkValidServiceWorker(url, config);

        navigator.serviceWorker.ready.then(() => {
          console.info(
            'This web app is being served cache-first by a service worker. To learn more, visit https://bit.ly/CRA-PWA',
          );
        });
      } else {
        registerValidSW(url, config);
      }
    });
  }
};

export const unregister = () => {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.ready.then(registration =>
      registration.unregister(),
    ).catch(error => {
      console.error(error.message);
    });
  }
};

