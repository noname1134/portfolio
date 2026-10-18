def reverse_array(arr):
    reversed_arr = []
    for i in range(len(arr) - 1, -1, -1):
        reversed_arr.append(arr[i])
    return reversed_arr


if __name__ == "__main__":
    arr = [43, "what", 9, True, "cannot", False, "be", 3, True]
    print(f"Original Array: {arr}")
    print("Reversed Array Output:")
    print(reverse_array(arr))
