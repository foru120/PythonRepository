import time
import sys

try:
    f = open('D:/02.Python/ch13/song.txt')
except FileNotFoundError:
    print('no file')
    sys.exit(0)
else:
    try:
        for line in f:
            if 'end' in line:
                raise SystemExit
            print(line, end='')
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    finally:
        print('file close')
        f.close()