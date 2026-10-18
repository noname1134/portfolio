def print_matrix(matrix):
    result = []
    for row in matrix:
        for element in row:
            result.append(str(element))
    print(" ".join(result))


if __name__ == "__main__":
    matrix = [
        [1, 2],
        [3, 4],
        [5, 6]
    ]
    print("print_matrix(matrix) → Should print: 1 2 3 4 5 6")
    print("Actual Output:", end=" ")
    print_matrix(matrix)
