def zero_count(matrix):
    count = 0
    for row in matrix:
        for element in row:
            if element == 0:
                count += 1
    return count


if __name__ == "__main__":
    matrix = [
        [0, 1, 1],
        [0, 1, 0],
        [1, 0, 0]
    ]
    print(f"print(zero_count(matrix)) → Should print: 5")
    print(f"Actual Output: {zero_count(matrix)}")
