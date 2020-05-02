def meanDigits(num):
    string = str(num)
    strings = list(string)
    integers = list(map(int, strings))
    return sum(integers) / len(integers)

meanDigits(12)
meanDigits(80)
meanDigits(666)



