def find_successive_match(arr):
    if len(arr) < 2:
        return -1
    i = 1
    while i < len(arr):
        if arr[i] == arr[i - 1]:
            return i
        i += 1
    return -1

if __name__ == "__main__":
    arr1 = [True, False, False, True, True, False]
    arr2 = [True, False, True, False, True, True]
    arr3 = [True, False, True, False, True, False]

    print(f"{arr1} → returns {find_successive_match(arr1)}")
    print(f"{arr2} → returns {find_successive_match(arr2)}")
    print(f"{arr3} → returns {find_successive_match(arr3)}")