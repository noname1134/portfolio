# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def str_to_float(s):
    if is_number(s):
        s = float(s)
        return s
    else:
        return False

def get_input_num():
    while True:
        s = input("")
        num = str_to_float(s)
        if num:
            return num
        print("Not a number")

def average(num_1, num_2, num_3):
    return (num_1 + num_2 + num_3) / 3

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Find three numbers average")
    print("First number: ")
    input_num_1 = get_input_num()
    print(input_num_1[0])
    print("Second number: ")
    input_num_2 = get_input_num()
    print("Third number: ")
    input_num_3 = get_input_num()

    av = average(input_num_1, input_num_2, input_num_3)
    print("Final average: ", av)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
