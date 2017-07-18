# cgihttpServer.py
from http.server import CGIHTTPRequestHandler, HTTPServer
import sys
ip = '127.0.0.1'
port = 8000
addr = (ip, port)

httpd = HTTPServer(addr, CGIHTTPRequestHandler)
servip, servport = httpd.socket.getsockname()
print("Serving HTTP on {}, port {}...".format(servip, servport))
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    httpd.server_close()
    sys.exit(0)
