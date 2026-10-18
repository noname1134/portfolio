def sum_two_arrays(arr1, arr2):
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result


if __name__ == "__main__":
    first_array = [4, 6, 7]
    second_array = [8, 1, 9]
    print(f"first_array = {first_array}")
    print(f"second_array = {second_array}")
    print(f"Function output: {sum_two_arrays(first_array, second_array)}")
