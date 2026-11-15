import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

CARD = """<article class="product_pod"><h3><a href="{slug}/index.html" title="{title}">{title}</a></h3>
<p class="star-rating Three"></p><div><p class="price_color">£{price}</p>
<p class="instock availability"> In stock </p></div></article>"""

DETAIL = """<html><body><ul class="breadcrumb"><li><a>Home</a></li><li><a>Books</a></li>
<li><a>Poetry</a></li><li>{title}</li></ul><div class="product_main"><h1>{title}</h1></div>
<div id="product_description"></div><p>A lovely description.</p>
<table class="table table-striped"><tr><th>UPC</th><td>upc-{slug}</td></tr>
<tr><th>Availability</th><td>In stock (7 available)</td></tr>
<tr><th>Number of reviews</th><td>0</td></tr></table></body></html>"""


def listing(page, books, last):
    cards = "".join(CARD.format(slug=s, title=t, price=p) for s, t, p in books)
    nxt = "" if last else f'<ul class="pager"><li class="next"><a href="page-{page + 1}.html">next</a></li></ul>'
    return f'<html><body><section>{cards}</section>{nxt}</body></html>'


class State:
    hits = defaultdict(int)
    broken_details = set()   # slugs whose detail page always returns 500


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="text/html; charset=utf-8", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client gave up (e.g. timeout test)

    def do_GET(self):
        path = self.path
        State.hits[path] += 1
        n = State.hits[path]

        if path == "/robots.txt":
            return self._send(200, b"User-agent: *\nDisallow: /private\n", "text/plain")
        if path == "/private":
            return self._send(200, b"<html>secret</html>")
        if path == "/flaky":                       # 503 twice, then OK
            return self._send(503) if n <= 2 else self._send(200, b"<html>ok</html>")
        if path == "/ratelimited":                 # 429 once with Retry-After
            return self._send(429, headers={"Retry-After": "0"}) if n == 1 else self._send(200, b"<html>ok</html>")
        if path == "/always503":
            return self._send(503)
        if path == "/forbidden":
            return self._send(403)
        if path == "/image":
            return self._send(200, b"\x89PNG", "image/png")
        if path == "/empty":
            return self._send(200, b"   ")
        if path == "/slow":
            time.sleep(2)
            return self._send(200, b"<html>late</html>")
        if path == "/ok":
            return self._send(200, "<html><body>£ ok</body></html>".encode("utf-8"), "text/html")

        #catalogue
        if path == "/catalogue/page-1.html":
            return self._send(200, listing(1, [("book-a", "Book A", "10.50"), ("book-b", "Book B", "20.00")], False).encode())
        if path == "/catalogue/page-2.html":
            return self._send(200, listing(2, [("book-c", "Book C", "5.25"), ("book-a", "Book A", "10.50")], False).encode())
        if path == "/catalogue/page-3.html":
            return self._send(404)
        for slug in ("book-a", "book-b", "book-c"):
            if path == f"/catalogue/{slug}/index.html":
                if slug in State.broken_details:
                    return self._send(500)
                title = slug.replace("-", " ").title()
                return self._send(200, DETAIL.format(title=title, slug=slug).encode())
        self._send(404)


def start():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"
