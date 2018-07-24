# Error8.py
while True:
    try:
        data = input("?>")
        print(10 / int(data))
    except(ZeroDivisionError, ValueError, KeyboardInterrupt) as e:
        print(e)
        if isinstance(e, KeyboardInterrupt):
            print("KeboardInterrupt")
            break
print("bye~")
