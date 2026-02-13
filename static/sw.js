self.addEventListener('install', (e) => {
  console.log('[Service Worker] Install');
});

self.addEventListener('fetch', (e) => {
  // Estrategia de red por defecto
  e.respondWith(fetch(e.request));
});
