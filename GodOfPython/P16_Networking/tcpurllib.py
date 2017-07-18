from urllib import request

# f = request.urlopen('http://www.daum.net')
# print(f.read(470).decode('utf-8'))
# meta = f.info()
# print(meta.as_string())

urlinst = request.Request('http://www.naver.com')
f = request.urlopen(urlinst)
print(f.read(100).decode('utf-8'))