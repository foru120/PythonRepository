import math

# 최대 공약수
def gcd_list(*n):
    def gcd(a):
        b = gcdtwo(max(a), min(a))
        a.remove(min(a))
        a.remove(max(a))
        a.append(b)
        if max(a) == min(a):
            print('최대공약수는 : ', a[0])
        else:
            gcd(a)

    def gcdtwo(a, b):
        if min(a, b) == 0:
            return max(a, b)
        return gcdtwo(b, a % b)

    a = []
    for i in n:
        a.append(i)
    gcd(a)


# 표준 편차
def stddev(*args):
    def mean():
        return sum(args) / len(args)

    def variance(m):
        total = 0
        for arg in args:
            total += (arg - m) ** 2
        return total / (len(args)-1)

    v = variance(mean())
    return math.sqrt(v)