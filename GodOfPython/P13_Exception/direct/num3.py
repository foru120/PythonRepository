import sys, time
num = 1

try:
    while num<=10:
        print(num)
        num += 1
        time.sleep(1)
except KeyboardInterrupt:
    print('exit')
    sys.exit(0)
else:
    print('complete')
finally:
    print('Goodbye Python')    