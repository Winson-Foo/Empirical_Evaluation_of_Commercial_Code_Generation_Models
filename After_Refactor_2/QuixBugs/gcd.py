# This function returns the greatest common divisor (GCD) of two numbers a and b
def gcd(a, b):
    # if b is 0, a is the GCD
    if b == 0:
        return a
    # otherwise, recurse on the remainder of a divided by b
    else:
        return gcd(b, a % b)