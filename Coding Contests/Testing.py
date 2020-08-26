def reverse(num):
    ls = list(str(num))
    ls = ls[::-1]
    rev = "".join(ls)
    return rev

reverse(-123)

