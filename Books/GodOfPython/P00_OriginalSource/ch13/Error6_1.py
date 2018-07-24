# Error6_1.py
dic = {'apple': 2, 'banana': 10, 'fineapple': 5}

while True:
    data = input(">")
    try:
        dic[data]
    except KeyError:
        print('There is no data.')
    except KeyboardInterrupt:
        break
    else:
        print("{} : {}개".format(data, dic[data]))

    print("continue...")
