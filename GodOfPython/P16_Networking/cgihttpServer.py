from http.server import CGIHTTPRequestHandler, HTTPServer
import sys

ip = '127.0.0.1'
port = 8000
addr = (ip, port)

httpd = HTTPServer(addr, CGIHTTPRequestHandler)
sevip, sevport = httpd.socket.getsockname()
print('Serving HTTP on {}, port {}...'.format(sevip, sevport))

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    httpd.server_close()
    sys.exit(0)