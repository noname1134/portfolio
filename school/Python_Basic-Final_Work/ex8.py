student = {
    'name': 'John',
    'age': 20,
    'hobbies': ['reading', 'games', 'coding'],
}


def print_student_data(s):
    for key, value in s.items():
        print(f"{key}: {value}")


def add_hobby(s, hobby):
    if hobby not in s['hobbies']:
        s['hobbies'].append(hobby)


def remove_hobby(s, hobby):
    if hobby in s['hobbies']:
        s['hobbies'].remove(hobby)

if __name__ == "__main__":
    print("=== Initial Student Data ===")
    print_student_data(student)

    print("\n=== After Adding Hobby 'sports' ===")
    add_hobby(student, 'sports')
    print_student_data(student)

    print("\n=== After Deleting Hobby 'games' ===")
    remove_hobby(student, 'games')
    print_student_data(student)

    print("\n=== After Adding 'family_name' Property ===")
    student['family_name'] = 'Doe'
    print_student_data(student)