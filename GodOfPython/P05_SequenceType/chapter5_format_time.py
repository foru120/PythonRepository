import time
now = time.localtime()
print("현재 시각은 %d년 %d월 %d일 %d시 %d분 %d초 입니다." %(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

print('{mon}월 {day}일 {hour}시'.format(mon=now.tm_mon,day=now.tm_mday,hour=now.tm_hour))