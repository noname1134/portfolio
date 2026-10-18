def get_user_data():
    full_name = ""
    while True:
        full_name = input("Enter your full name (First and Last name): ").strip()
        words = full_name.split()
        if len(words) == 2 and all(w.isalpha() for w in words):
            break
        print("Invalid input. Please enter exactly two alphabetical words.")

    age = None
    while True:
        age_input = input("Enter your age (1-130): ").strip()
        if age_input.isdigit():
            age = int(age_input)
            if 1 <= age <= 130:
                break
        print("Invalid input. Please enter a valid number between 1 and 130.")

    email = ""
    while True:
        email = input("Enter your email: ").strip()
        if "@" in email and len(email) > 1 and not email.startswith("@") and not email.endswith("@"):
            break
        print("Invalid input. Email must contain '@' properly.")

    print("\n--- Registration Successful ---")
    print(f"Full Name: {full_name}")
    print(f"Age: {age}")
    print(f"Email: {email}")

if __name__ == "__main__":
    get_user_data()