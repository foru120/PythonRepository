def euclidean_gcd(num1, num2):
    rem = max(num1, num2) % min(num1, num2)
    if rem != 0:
        return euclidean_gcd(num1, rem)
    return min(num1, num2)

print(euclidean_gcd(108, 72))

def prime_factorization_gcd(num_list):
    minNum = min(num_list)
    for i in range(2, minNum+1):
        cnt = 0
        for num in num_list:
            if (num % i) == 0:
                cnt += 1
        if len(num_list) == cnt:
            return i*prime_factorization_gcd(list(int(num/i) for num in num_list))
    return 1

print(prime_factorization_gcd([108, 72, 24]))