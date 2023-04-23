import string

def to_base(num, b):
    """
    Converts a decimal integer to its representation in base b.
    Returns the resulting string.
    """
    result = ''  # initialize result string
    alphabet = string.digits + string.ascii_uppercase  # define alphabet for conversion

    # loop through digits of num from most significant to least significant
    while num > 0:
        remainder = num % b  # find remainder of num / b
        quotient = num // b  # compute integer quotient of num / b
        digit = alphabet[remainder]  # map remainder to appropriate digit in the alphabet
        result = digit + result  # add newly computed digit to the front of the result string
        num = quotient  # set num to quotient for next iteration of loop

    return result[::-1]  # reverse the order of the characters in the result string