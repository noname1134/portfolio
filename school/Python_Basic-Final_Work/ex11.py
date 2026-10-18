def find_dup(arr):
    counts = {}
    for item in arr:
        counts[item] = counts.get(item, 0) + 1

    duplicates = []
    for item, count in counts.items():
        if count > 1:
            duplicates.append(item)
    return duplicates


if __name__ == "__main__":
    arr = [4, 2, 34, 4, 1, 12, 1, 4]
    print(f"arr = {arr}")
    print(f"print(find_dup(arr)) → Should print: [4, 1]")
    print(f"Actual Output: {find_dup(arr)}")

