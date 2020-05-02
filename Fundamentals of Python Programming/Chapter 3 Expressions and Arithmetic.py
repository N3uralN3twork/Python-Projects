"""Chapter 3: Expressions and Arithmetic

What You'll Learn:
Expressions
Mixed Type Expressions
Operator Precedence and Associativity
Formatting Expressions
Comments
Errors
More Arithmetic Operators
Algorithms"""

"3.1 Expressions"
-------------------------
X+Y| x added to y
-------------------------
X-Y| x subtracted from y
-------------------------
X*Y| x times y
-------------------------
X/Y| x divided by y
-------------------------
X%Y| Remainder of x divided by y
-------------------------
X**Y| X raised to the power of y
-------------------------

# All expressions have some sort of value
# It should be noted that fractions in binary behave differently than expected
# Use integers to count things and floating point numbers to measure other things

"3.2 Mixed Type Expressions"
x = 4
y = 10.2
sum = x+y
type(sum) # returns a floating point number
# Integer + Integer = Integer
# Float + Float = Float
# Integer + Float = Float

"3.3 Operator Precedence and Associativity"
# Precedence = with two different operators, which goes first?
# Associativity = with two operators of same precedence, which goes first?

# Examples:
2 + 3 * 4 # 14
# is interpreted as:
(2 + 3) * 4

# Multiplication and Division are done before Addition and Subtraction in Python
# You can use parentheses to override precedence

# Chained assignemnt:
x = y = 2


"3.4 Formatting Expressions"
# Python has no implicit multiplication
# Example:
    2y
# Thus, you must explicitly state that you want to multiply
2*y
# Other people will examine your code, so readability is important


"3.5 Comments"
# Good programmers annotate their code by inserting remarks that explain the code
# Any text contained within comments is ignored by the Python interpreter
# You should understand comments by now geez...

"3.6 Errors"

# 3 Types of errors:
    # Syntax, Run-time, Logic
    # Also called bugs

# Syntax errors:
    # Code won't run with syntax errors present
    # Detected immediately
    # Poor assignments
    # Misspellings
    # Parentheses mismatch
    # Quote mismatches
    # Poor indentation
x = )3 + 4()
x = 'Hello"

# Run-time errors:
    # Occur during the execution phase
    # Don't always manifest themselves immediately
    # Can still execute code sometimes

print("Bobo")/2

# Logic errors:
    # Program runs, but produces the wrong output
    # Interpreter does not provide location of logic errors
    # Thus, they tend to be the most difficult to fix

"3.7 Arithmetic Examples"

# Increment statement (+=)
x += 1

"x op=exp",
    # x = variable
    # op = an arithmetic operator
    # exp = some compatible expression
# Examples
x += 1 # Add 1 to x
y *= x # Multiply y by x
x /= y

"3.8 Algorithms"
# A finite sequence of steps that solves a problem or produces a result
# Examples:
    # Recipes
    # Relationships between things

















