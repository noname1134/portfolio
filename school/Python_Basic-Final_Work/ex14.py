def check_palindrome(word):
    cleaned = "".join(word.lower().split())
    return cleaned == cleaned[::-1]


if __name__ == "__main__":
    first_str = "racecar"
    second_str = "Java"

    print(f"first_str = '{first_str}' -> {check_palindrome(first_str)}")
    print(f"second_str = '{second_str}' -> {check_palindrome(second_str)}")
