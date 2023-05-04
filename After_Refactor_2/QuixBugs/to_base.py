import string

def convert_to_base(number, base):
    # Define the alphabet for the given base
    alphabet = string.digits + string.ascii_uppercase
    
    # Initialize variables for the converted number and remainder
    converted_number = ''
    remainder = 0
    
    # Convert the number to the given base
    while number > 0:
        remainder = number % base
        number = number // base
        converted_number = alphabet[remainder] + converted_number
    
    # Return the converted number
    return converted_number 