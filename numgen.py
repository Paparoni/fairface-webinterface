from cryptography.fernet import Fernet
import datetime

def generate():
    # Generate a key for cryptography
    key = Fernet.generate_key()

    # Get the current date and time
    now = datetime.datetime.now()

    # Convert the current date and time to bytes
    now_bytes = now.strftime("%Y-%m-%d %H:%M:%S.%f").encode()

    # Encrypt the current date and time with cryptography
    f = Fernet(key)
    encrypted_now = f.encrypt(now_bytes)

    # Convert the encrypted current date and time to an integer
    encrypted_now_int = int.from_bytes(encrypted_now, byteorder="big")

    # Get the last 5 digits of the encrypted current date and time
    encrypted_now_int_str = str(encrypted_now_int)
    last_5_digits = encrypted_now_int_str[-5:]

    # Convert the last 5 digits to an integer
    five_digit_int = int(last_5_digits)

    return five_digit_int

