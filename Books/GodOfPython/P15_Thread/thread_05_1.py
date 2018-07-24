import threading

Q=1000000
thread_list = []

def drink(max):
    global Q

    for i in range(0, max):
        Q-=1

for i in range(0, 2):
    thread_inst = threading.Thread(target=drink, args=(500000,))
    thread_list.append(thread_inst)
    thread_inst.start()

for thread in thread_list:
    thread.join()

print(Q)