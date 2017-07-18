# Error7.py
import time
import sys


try:
    f = open("C:/gop/ch13/song.txt")
except FileNotFoundError:
    print("no file")
    sys.exit(0)
else:
    try:
        for line in f:
            if 'end' in line:  # 파일에서 읽어들인 라인에 'end'라는 단어가 있다면
                raise SystemExit  # 프로그램 종료
            print(line, end='')
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:  # 반드시 실행되어야 할 코드(finally)
        print("file close")
        f.close()  # 파일을 닫는다.
