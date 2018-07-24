def factorial2(num):
    if num > 1:
        num *= factorial2(num-1)
    return num

print(factorial2(10))