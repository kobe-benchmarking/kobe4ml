import http.server
import ssl
import json
import base64
from functools import partial

with open("credentials.json", "r") as f:
    creds = json.load(f)

USERNAME = creds["username"]
PASSWORD = creds["password"]

class AuthHandler(http.server.SimpleHTTPRequestHandler):
    def do_AUTHHEAD(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="Protected"')
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        auth_header = self.headers.get('Authorization')
        expected_auth = 'Basic ' + base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()

        if auth_header == expected_auth:
            super().do_GET()
        else:
            self.do_AUTHHEAD()
            self.wfile.write(b'Authentication required.')

if __name__ == "__main__":
    PORT = 48031
    DIRECTORY = "/home/ec2-user/webshare/content"

    handler_class = partial(AuthHandler, directory=DIRECTORY)

    server_address = ('0.0.0.0', PORT)
    httpd = http.server.HTTPServer(server_address, handler_class)
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile='server.pem')
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"Serving HTTPS on port {PORT}")
    httpd.serve_forever()