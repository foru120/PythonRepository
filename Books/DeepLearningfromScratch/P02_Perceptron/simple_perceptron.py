def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2

    return 1 if tmp > theta else 0

print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))