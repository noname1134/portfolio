def remove_duplicates_skip_pete(arr):
    result = []
    i = 0
    while i < len(arr):
        item = arr[i]
        if isinstance(item, str) and item.lower() == 'pete':
            i += 1
            continue

        exists = False
        j = 0
        while j < len(result):
            if result[j] == item:
                exists = True
                break
            j += 1

        if not exists:
            result.append(item)
        i += 1
    return result


if __name__ == "__main__":
    names = ['Chris', 'Kevin', 'Naveed', 'Pete', 'Victor', 'Chris', 'Kevin']
    print(f"Input: {names}")
    print(f"Output: {remove_duplicates_skip_pete(names)}")

