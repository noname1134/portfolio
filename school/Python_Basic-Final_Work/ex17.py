def find_duplicates_while(arr):
    duplicates = []
    i = 0
    while i < len(arr):
        item = arr[i]
        count = 0
        j = 0
        while j < len(arr):
            if arr[j] == item:
                count += 1
            j += 1

        if count > 1:
            already_added = False
            k = 0
            while k < len(duplicates):
                if duplicates[k] == item:
                    already_added = True
                    break
                k += 1
            if not already_added:
                duplicates.append(item)
        i += 1
    return duplicates


if __name__ == "__main__":
    names = ['Chris', 'Kevin', 'Naveed', 'Pete', 'Victor', 'Chris', 'Kevin']
    print(f"Input: {names}")
    print(f"Output: {find_duplicates_while(names)}")
