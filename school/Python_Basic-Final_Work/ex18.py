def remove_duplicates_while(arr):
    unique_arr = []
    i = 0
    while i < len(arr):
        item = arr[i]
        exists = False
        j = 0
        while j < len(unique_arr):
            if unique_arr[j] == item:
                exists = True
                break
            j += 1
        if not exists:
            unique_arr.append(item)
        i += 1
    return unique_arr


if __name__ == "__main__":
    names = ['Chris', 'Kevin', 'Naveed', 'Pete', 'Victor', 'Chris', 'Kevin']
    print(f"Input: {names}")
    print(f"Output: {remove_duplicates_while(names)}")