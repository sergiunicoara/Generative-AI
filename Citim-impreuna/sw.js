const CACHE = "citim-impreuna-v20";
const ASSETS = [
  ".",
  "index.html",
  "css/style.css",
  "js/app.js",
  "js/config.js",
  "js/scenes.js",
  "js/tracker.js",
  "js/verses-1samuel.js",
  "js/verses-2samuel.js",
  "js/verses.js",
  "manifest.webmanifest",
  "icons/icon.svg",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  // Doar fișierele proprii (GET); apelurile către Supabase trec neatinse
  // (cache.put nu acceptă POST, iar statisticile nu trebuie servite din cache).
  if (event.request.method !== "GET" || !event.request.url.startsWith(self.location.origin)) {
    return;
  }
  // Rețea întâi, ca actualizările să apară imediat; recurge la cache doar offline.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(event.request, copy));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
