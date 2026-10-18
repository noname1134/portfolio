def get_arr_sum(arr: list) -> int:
    arr_sum = 0
    for num in arr:
        arr_sum += num

    return arr_sum

if __name__ == "__main__":
    arr = [1,13,22,123,49,34,5,24,57,45]
    arr_sum = get_arr_sum(arr)
    print(arr_sum)