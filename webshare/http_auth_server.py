import http.server
from functools import partial

if __name__ == "__main__":
    PORT = 48031
    DIRECTORY = "/home/ec2-user/kobe4ml/webshare/content"

    handler_class = partial(http.server.SimpleHTTPRequestHandler, directory=DIRECTORY)
    server_address = ('0.0.0.0', PORT)
    httpd = http.server.HTTPServer(server_address, handler_class)

    print(f"Serving HTTP (no SSL, no auth) on port {PORT}")
    httpd.serve_forever()