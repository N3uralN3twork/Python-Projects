###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python-Projects/Coding Contests")
os.chdir(abspath)
###############################################################################
###                     2. Coding Contests                                  ###
###############################################################################


a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

# Only print numbers less than 5

b = [i for i in a if i < 5]

#or

for number in range(len(a)):
    if a[number] < 5:
        print(a[number])
    else:
        pass

# Make a list of only first and lasat elements in a list

def fl(lst):
    return [lst[0], lst[-1]]


fl(a)


# Fibonacci numbers where input is the number of elements in the sequence

def Fibo(n):
    if n <= 2:
        return 1
    else:
        return Fibo(n - 1) + Fibo(n - 2)


Fibo(3)


# Print the number of vowels in a string
def count_vowels(txt):
    vowels = ["a", "A", "e", "E", "i", "I", "o", "O", "u", "U"]
    count = 0
    for letter in txt:
        if letter in vowels:
            count = count + 1
        else:
            pass
    print(f"Number of vowels: {count}")


text1 = "Hello"
text2 = "Hello World"

count_vowels(txt=text1)
count_vowels(txt=text2)


# Return the factorial of a given integer

def factorial(N):
    if N <= 1:
        return 1
    else:
        return N * factorial(N - 1)


factorial(0)
factorial(1)
factorial(2)
factorial(3)


# Given a list of numbers, return a list that has no duplicates and it sorted from least to greatest value.

def unique_sort(lst):
    sett = set(lst)  # To remove duplicates, use a set().
    result = list(sett)  # Convert to list
    result.sort()  # Sort the list
    return result


a = [1, 3, 4, 5, 6, 100, 27]
unique_sort(lst=a)


# Given the height and width, return the perimeter of a rectange
def perimeter(height, width):
    return (2 * height) + (2 * width)


perimeter(20, 10)


# Given string, return string in alphabetical order

def alphabet_soup(text):
    lst = tuple(text)
    lst = sorted(lst)
    string = "".join(lst)
    return string


alphabet_soup("hello")
alphabet_soup("geek")


# Given a grade string, determine whether the student passed or failed

def grade_percentage(user_score: str, pass_score: str):
    user_score = int(user_score.strip("%"))
    pass_score = int(pass_score.strip("%"))
    if user_score >= pass_score:
        print("You PASSED the Exam")
    else:
        print("You FAILED the Exam")


grade_percentage("85%", "85%")
grade_percentage("65%", "90%")


def next_in_line(lst: list, num: int):
    if len(lst) != 0:
        lst.append(num)  # Add the number to the end of the list
        lst.pop(0)  # Remove the first element of a list
        return lst
    else:
        return "No list has been selected"


next_in_line([5, 6, 7, 8, 9], 1)
next_in_line([], 6)


def sum_natural(n: int) -> int:
    if n <= 1:
        return n
    else:
        return n + sum_natural(n - 1)


sum_natural(12)
sum_natural(5)


# Given a list of integers between 1 and N, find the one missing number in the sequence
# Sum of whole list = n(n+1)/2
# Find sum of actual list
# Subtract to find the missing number

def missing_num(lst: list) -> int:
    N = max(lst)
    theory = (N * (N + 1)) / 2
    actual = sum(lst)
    MissingNum = theory - actual
    return int(MissingNum)


missing_num([1, 2, 3, 5])


def apocalyptic(N: int):
    num = str(2 ** N) # Calculate the number 2^n
    if "666" in num:  # Check for 666 substring
        position = num.find("666") # Return the position of the substring
        print(f"Repent! {position} days until the Apocalypse!")
    else: # Otherwise, if it's not True
        print("Crisis averted. Resume sinning")


apocalyptic(499)
apocalyptic(157)
apocalyptic(175)
apocalyptic(220)


def diff_max_min(lst: list)-> int:
    Min = min(lst) # Find the minimum of the list
    Max = max(lst) # Find the maximum of the list
    return Max - Min # Return the difference


diff_max_min([10, 4, 1, 4, -10, -50, 32, 21])
diff_max_min([44, 32, 86, 19])


def remove_enemies(names: list, enemies: list)-> list:
    return [i for i in names if i not in enemies]


remove_enemies(["Fred"], [])
remove_enemies(["Adam", "Emmy", "Tanya", "Emmy"], ["Emmy"])


def only_integers(lst: list):
    return [x for x in lst if type(x) is int]


a = [9, 2, "space", "car", "lion", 16, 3.3]
b = ["String", True, 3.3, 1]

only_integers(a)
only_integers(b)

# Maximum Difference

def difference(lst: list)-> int:
    return max(lst) - min(lst)


difference([10, 15, 20, 2, 10, 6])

def reverse(arg):
    if arg == True and type(arg) == bool:
        return False
    elif arg == False and type(arg) == bool:
        return True
    else:
        return "boolean expected"

reverse(True)
reverse(False)
reverse(0)
reverse(None)


# Number of solutions to quadratic equation

def solutions(a, b, c):
    discrim = (b**2)-(4*a*c)
    if discrim > 0:
        return 2
    elif discrim == 0:
        return 1
    else:
        return 0

solutions(1, 0, -1)


# Filter a list for just numbers:

def filter_list(lst: list):
    output = [i for i in lst if type(i) is int]
    return output

list1 = [1, 2, "a", "b"]
list2 = [1, "a", "b", 0, 15]
list3 = [1, 2, "aasf", "1", "123", 123]

filter_list(list1)
filter_list(list2)
filter_list(list3)

# Concatenate the elements of two lists:

a = [1, 2]
b = [3, 4]
c = a+b

# Stupid Addition:

def stupid_addition(arg1, arg2):
    if type(arg1) != type(arg2):
        return None
    elif type(arg1) is str and type(arg2) is str:
        return int(arg1) + int(arg2)
    else:
        return str(arg1) + str(arg2)

stupid_addition(1, 2)
stupid_addition("1", "2")
stupid_addition("1", 2)


# Sort a list by String Length:
def sort_by_length(lst):
    return (sorted(lst, key=len))

list1 = ["Google", "Apple", "Microsoft"]
sort_by_length(list1)


# Multiply List Elements by its Length


def MultiplyByLength(lst: list):
    length = len(lst)
    return [length*i for i in lst]

lst1 = [2, 3, 1, 0]
lst2 = [4, 1, 1]

MultiplyByLength(lst1)
MultiplyByLength(lst2)


# Is a number symmetrical?

def is_symmetrical(num: int):
    string = str(num)[::-1] # To reverse a string
    backwards = int(string)
    if backwards == num:
        return True
    else:
        return False

is_symmetrical(123)
is_symmetrical(7227)


# Mean of all digits in a number

def meanDigits(num):
    string = str(num)
    strings = list(string)
    integers = list(map(int, strings))
    return sum(integers) / len(integers)

meanDigits(12)
meanDigits(80)
meanDigits(666)


# Equality of three values

def equal(a, b, c):
    return {3:0, 2:2, 1:3}[len({a, b, c})]

equal(3,4,3)


# The Karaca's Encryption Algorithm - Hard

def karaca(word: str)-> str:
    strReverse = word[::-1] # Reverse the string
    vowel_dict = {'a':'0', 'e':'1', 'o':'2', 'u':'3'}
    for item in strReverse:
        if item in vowel_dict.keys():
            string = strReverse.replace(item, vowel_dict[item])
    return string + "aca"

karaca("banana")
karaca("apple")
karaca("karaca")
karaca("burak")


# List of Multiples

def list_of_multiples(num, length):
    return [(i+1)*num for i in range(length)]

list_of_multiples(7, 5)




# Calculate the profit:

def profit(info: dict):
    cost, sell, amount = info["cost_price"], info["sell_price"], info["inventory"]
    return round(((sell*amount) - (cost*amount)), 0)


info = {
  "cost_price": 32.67,
  "sell_price": 45.00,
  "inventory": 1200
       }

profit(info)

# Remove Strings from a list:

def filter_list(lst: list)->list:
    return [i for i in lst if type(i) is int]

list1 = [1, 2, 3, "a", "b", 4]

filter_list(list1)


# Even Number Generator
def find_even_nums(num: int):
    nums = list(range(1, num+1))
    return [i for i in nums if i%2 == 0]

find_even_nums(8)


# Sum of positive integers


def positive_sum(array: list):
    positives = [i for i in array if i>0]
    return sum(positives)

list1 = [1, -4, 7, 12]
positive_sum(list1)

# Shortest Word in string


def find_short(string):
    words = string.split()
    lengths = map(len, words)
    return min(lengths)

find_short("Hello there dude")


def high_and_low(numbers: str):
    numbers = numbers.split()
    numbers = [int(i) for i in numbers]
    minimum = min(numbers)
    maximum = max(numbers)
    print(f"'{maximum} {minimum}'")

high_and_low("4 5 29 54 4 0 -214 542 -64 1 -3 6 -6")

# Extract the century from the year
def century(year: int):
    return (year) // 100 + 1

century(1900)


# Remove vowels from a string

def disemvowel(string: str):
    vowels = ["a", "A", "e", "E", "i", "I", "o", "O", "u", "U"]
    letters = list(string)
    novowels = [i for i in letters if i not in vowels]
    result = "".join(novowels)
    return result

disemvowel("Hello")


# Abbreviate a Two Word Name

def abbrevName(name: str):
    parts = name.split()
    FirstOne = name[0] # First Word First Letter
    LastOne = parts[1][0] # Second Word First Letter
    FirstOne = FirstOne.upper()
    LastOne = LastOne.upper()
    return FirstOne + "." + LastOne

name = "Same Harris"
name[0]
abbrevName("tiffany Yulish")

# Array plus Array:

def array_sum(arr1, arr2):
    return sum(arr1+arr2)


# Is he gonna survive?

def hero(bullets, dragons):
    if dragons == 0:
        return True
    else:
        return True if bullets/dragons >= 2 else False
hero(7, 4)
hero(200, 20)


# Return the first M multiples of N
def multiples(m, n):
    n = int(n)
    nums = list(range(1, (m*n+1)))
    return [float(i) for i in nums if i%n == 0]

multiples(3,5.0)


def check(seq, elem):
    return True if elem in seq else False

# Angle of right triangle
def right_angle(AB:int, BC:int):
    div = AB/BC
    import math
    return math.degrees(math.atan(div))

right_angle(10,10)

# (a^b)+(c^d)
def integer_powers(a,b,c,d):
    print((a**b)+(c**d))


# Find the missing number in a list of integers:
    # Common algorithm

def missing_num(lst: list)->int:
    n = len(lst)
    total = ((n+1)*(n+2))/2
    sumList = sum(lst)
    return total - sumList

missing_num([1,2,4])

# Given an amount, what is the minimum # of coins needed to pay for it?
def ATM(n):
    coins = [1, 5, 10, 25]
    count = 0
    if n == 0:
        return 0
    else:


ATM(3)
# I have an idea of how to complete this problem, but I'm just not sure
# how to implement it.



# Print the single unique number in a given list:
def unique_num(lst:list)->list:
    return [i for i in lst if lst.count(i) == 1]

list1 = [1, 2, 1, 1]

unique_num(list1)

# Print out any duplicate variables:
def duplicate_vars(lst:list):
    predictors = [var for var in lst if lst.count(var) >= 2]
    return set(predictors)
variables = ["Age", "Sales", "Company", "Age"] # Age is the duplicate variable


duplicate_vars(variables)





