"""Chapter 2: Values and Variables

Numeric values
Strings
Variables
Assignment
Identifiers
Reserved words

To delete some variable:

del variable1, variable2, ...
"""

"2.1 Integer and String Values"
# Ex: 4 = integer number

3+4 # Normal arithmetic addition

# 2,468 would appear as 2468, or without the comma

# String = a sequence of characters
# Delimited by single or double quotes ('', "")
# Ex:
"Hello"
# Interpreters always output a string in single quotes

# What if you're missing a matching quotation mark?

"Hello # You'll get an error here

"2.2 Variables and Assignment"

# Assignment statement:
    # associates a value with a variable
    # = is called the assignment operator
    # you can change the value of a same variable later on


x = 10 # Assign the numeric value of 10 to the variable x
print(x)

# You can assign multiple variables in one statement using a tuple assignment
x, y, z = 1, 2, 3

# A tuple is a comma-separated list of expressions
# Only works if both sides contain the same number of elements
x, y = 1, 2, 3


"2.3 Identifiers"
# Identifier = a word used to name things
# Rules:
    # Must contain at least one character
    # No spaces allowed
    # Reserved words are not allowed
    # Cannot begin with an integer

"2.4 Floating-point Numbers"
# Floating-point number = non-integers, with decimals
float(123/3)
# Does not allow for infinite decimal places, like pi (3.14)
round(123.334, 2)


"2.5 Control Codes within Strings"
# Special characters within strings that do something
# Denoted by "\", a backslash

# Control Codes:
    # \n = newline
    # \t = new tab
    # \b = backspace
    # \\ = backslash
# Examples:
print("Hello \nWorld")
print("Hello \tWorld")
print("Hello \bWorld")
print("Hello \\World")


"2.8 String Formatting"
# print(f"{})
# thing inside brackets is called the positional parameter

# Example:
name = "Matthias"
age = 22
print(f"Hello {name}")
print("Hello {}".format(name))
print("Hello {} who is {}".format(name, age))

"2.9 Multi-line Strings"
# Denoted by the triple string ''', or, """
# Example
"""Hello
How are you today"""
# is the same thing as:
print("Hello \nHow are you today")


"Exercises"
8.
# Yes, you can assign more than one variable in a single statement with a tuple

17.
print("A\nB\nC\nD\nE\nF")