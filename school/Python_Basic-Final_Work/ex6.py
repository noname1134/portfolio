def remove_property_from_students(students, property_name):
  """Checks for each property in each object and deletes it if it exists."""
  for student in students:
    if property_name in student:
      student.pop(property_name)
  return students


def print_student_properties(students):
  """Prints each property and value of each student object."""
  for index, student in enumerate(students, start=1):
    print(f"--- Student {index} ---")
    for key, value in student.items():
      print(f"{key}: {value}")
    print()


def sort_students_by_age_descending(students):
  """Sorts the array by age from oldest to youngest and returns the sorted array."""
  return sorted(students, key=lambda s: s["age"], reverse=True)


def main():
  students = [
      {
          "id": 1,
          "first_name": "John",
          "last_name": "Doe",
          "age": 20,
          "country": "USA",
          "city": "New York",
      },
      {
          "id": 2,
          "first_name": "Anna",
          "last_name": "Taylor",
          "age": 25,
          "country": "UK",
          "city": "London",
      },
      {
          "id": 3,
          "first_name": "Carlos",
          "last_name": "Gomez",
          "age": 22,
          "country": "Spain",
          "city": "Madrid",
      },
  ]
  print("=== Initial Students List ===")
  print_student_properties(students)

  print("=== After Removing 'country' Property ===")
  remove_property_from_students(students, "country")
  print_student_properties(students)

  print("=== Sorted by Age (Oldest to Youngest) ===")
  sorted_students = sort_students_by_age_descending(students)
  for s in sorted_students:
      print(s)


if __name__ == "__main__":
    main()