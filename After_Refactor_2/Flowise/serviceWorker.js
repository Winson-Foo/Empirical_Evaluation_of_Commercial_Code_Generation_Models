const isLocalhost = Boolean(
  window.location.hostname === 'localhost' ||
  window.location.hostname === '[::1]' || 
  window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/)
);

const registerValidServiceWorker = (swUrl, config) => {
  navigator.serviceWorker.register(swUrl).then(registration => {
    const installingWorker = registration.installing;
    if (!installingWorker) return;

    installingWorker.onstatechange = () => {
      if (installingWorker.state === 'installed') {
        if (navigator.serviceWorker.controller) {
          console.info('New content is available and will be used when all tabs for this page are closed. See https://bit.ly/CRA-PWA.');
          if (config && config.onUpdate) config.onUpdate(registration);
        } else {
          console.info('Content is cached for offline use.');
          if (config && config.onSuccess) config.onSuccess(registration);
        }
      }
    };
  })
  .catch(error => console.error('Error during service worker registration:', error));
};

const checkValidServiceWorker = (swUrl, config) => {
  fetch(swUrl, {headers: {'Service-Worker': 'script'}})
  .then(response => {
    const contentType = response.headers.get('content-type');
    if (response.status === 404 || (contentType != null && contentType.indexOf('javascript') === -1)) {
      navigator.serviceWorker.ready.then(registration => registration.unregister().then(() => window.location.reload()));
    } else {
      registerValidServiceWorker(swUrl, config);
    }
  })
  .catch(() => console.info('No internet connection found. App is running in offline mode.'));
};

export const register = config => {
  if ('serviceWorker' in navigator) {
    const publicUrl = new URL(process.env.PUBLIC_URL, window.location.href);
    if (publicUrl.origin !== window.location.origin) return;
    window.addEventListener('load', () => {
      const swUrl = `${process.env.PUBLIC_URL}/service-worker.js`;
      if (isLocalhost) {
        checkValidServiceWorker(swUrl, config);
        navigator.serviceWorker.ready.then(() => console.info('This web app is being served cache-first by a service worker. To learn more, visit https://bit.ly/CRA-PWA'));
      } else {
        registerValidServiceWorker(swUrl, config);
      }
    });
  }
};

export const unregister = () => {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.ready.then(registration => registration.unregister())
    .catch(error => console.error(error.message));
  }
};