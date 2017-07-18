while True:
    try:
        data = input('>')
        print(10/int(data))
    except(ZeroDivisionError, ValueError, KeyboardInterrupt) as e:
        print(e)
        if isinstance(e, KeyboardInterrupt):
            print('KeyboardInterrupt')
            break
print('bye~')