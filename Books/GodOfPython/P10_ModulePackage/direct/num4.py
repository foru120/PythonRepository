import sys
numbers = sys.argv[1:]

sum=0
for n in numbers:
    sum+=int(n)
    
print(sum)