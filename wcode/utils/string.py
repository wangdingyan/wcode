import random
import string


def generate_random_string(length=8):
    letters = string.ascii_letters  # 包含所有的大小写字母
    random_string = ''.join(random.choice(letters) for i in range(length))
    return random_string


if __name__ == '__main__':
    print(generate_random_string())