"""Static file server for local dev with caching fully disabled.

Plain `python -m http.server` lets browsers cache files with no explicit
Cache-Control header, so an already-open tab can keep serving old JS/CSS
after edits even on a normal refresh (only a hard refresh / cleared cache
forces a re-fetch). This server sends Cache-Control: no-store on every
response so any reload of an open tab always gets the current file.
"""

import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Strip conditional-request headers so send_head() never short-circuits
        # to a 304 Not Modified based on the browser's locally cached copy.
        self.headers["If-Modified-Since"] = ""
        self.headers["If-None-Match"] = ""
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8321
    ThreadingHTTPServer(("", port), NoCacheHandler).serve_forever()
