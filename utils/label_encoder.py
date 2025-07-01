def encode_label(char):
    return ord(char.upper()) - ord('A')

def decode_label(index):
    return chr(index + ord('A'))
