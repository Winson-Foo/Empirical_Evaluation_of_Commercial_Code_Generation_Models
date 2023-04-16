def matrix_cipher_decrypt(ciphertext, matrix):
    """
    Decrypts the given ciphertext using the matrix cipher with the given key.
    """
    decrypted_text = ""
    for char in ciphertext:
        if char.isalpha():
            row = 0
            col = 0
            # Find the row and column of the character in the matrix
            for i in range(5):
                for j in range(5):
                    if matrix[i][j] == char:
                        row = i
                        col = j
                        break
                if matrix[i][j] == char:
                    break
            # Find the corresponding character in the matrix (the inverse mapping)
            decrypted_char = chr(65 + row*5 + col)
            decrypted_text += decrypted_char
        else:
            decrypted_text += char
    return decrypted_text



#######################################################################################

while True:
    # Prompt the user to choose a mode
    mode = input("Enter 'e' for encryption, 'd' for decryption, or 'q' to quit: ")
    
    # Quit the program if 'q' is entered
    if mode == 'q':
        print("Goodbye!")
        break
    
    # Encryption mode
    elif mode == 'e':
        # Prompt the user to enter the plaintext and key
        plaintext = input("Enter the plaintext to encrypt: ")
        key = int(input("Enter the key to use for encryption (caesar 1-26): "))

        # Encrypt the plaintext using the Caesar cipher with the given key
        ciphertext = caesar_cipher_encrypt(plaintext, key)

        # Prompt the user to enter the Vigenere key to use for encryption
        key2 = input("Enter the Vigenere key to use for encryption: ")

        # Encrypt the Caesar cipher output using the Vigenere cipher with the given key
        final_text = vigenere_cipher_encrypt(ciphertext, key2)

        # Prompt user to select a matrix cipher
        matrix = select_matrix_cipher()

        # Encrypt the final test output using matrix cipher
        encrypted_text = matrix_cipher_encrypt(final_text, matrix)

        # Display the final encrypted text
        print("Final text:", encrypted_text)
    
    # Decryption mode
    elif mode == 'd':
       # Prompt the user to enter the ciphertext and the Vigenere key used to encrypt it
        ciphertext = input("Enter the ciphertext to decrypt: ")

        # Prompt user select matrix cipher
        matrix = select_matrix_cipher()

        # Decrypt the ciphertext using matrix cipher
        decrypted_text = matrix_cipher_decrypt(ciphertext, matrix)

        # Prompt the user to enter the Vigenere key used for encryption
        key1 = input("Enter the Vigenere key to use for decryption: ")

        # Decrypt the ciphertext using the Vigenere cipher with the given key
        plaintext = vigenere_cipher_decrypt(decrypted_text, key1)

        # Prompt the user to enter the Caesar key used for encryption
        key2 = int(input("Enter the Caesar key used to encrypt the plaintext: "))

        # Decrypt the resulting plaintext using the Caesar cipher with the given key
        plaintext = caesar_cipher_decrypt(plaintext, key2)

        # Display the resulting plaintext
        print("The decrypted text is:", plaintext)

    # Invalid mode
    else:
        print("Invalid mode. Please enter 'e', 'd', or 'q'.")