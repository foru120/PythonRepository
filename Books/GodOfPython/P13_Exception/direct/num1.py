import time
import sys
num=1

while num<=10:
    try:
        print(num)
        time.sleep(1)
        num+=1
    except KeyboardInterrupt:
        print('exit')
        sys.exit(0)
