from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
import time

ip = '127.0.0.1'
port = 8000
addr = (ip, port)

class myHTTPHandle(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(bytes('<html><body><p>{}</p></body></html>'.format(time.ctime()), 'utf-8'))

httpd = HTTPServer(addr, myHTTPHandle)
servip, servport = httpd.socket.getsockname()
print('Serving HTTP on {}, port {}'.format(servip, servport))

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    httpd.server_close()
    sys.exit(0)