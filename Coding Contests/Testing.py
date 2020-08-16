n = int(input())
print(n)
while n > 1:
    if n%2 == 0: # Test if even
        n //= 2 # Divide by two and round to whole number
    else: # If odd
        n = (n*3)+1  # Multiply by 3 and add 1
    print(n)

