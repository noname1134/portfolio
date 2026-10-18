our_pets = [
    {
        "animal_type": "cat",
        "names": ["Meowzer", "Fluffy", "Kit-Cat"]
    },
    {
        "animal_type": "dog",
        "names": ["Spot", "Bowser", "Frankie"]
    }
]


def print_cat_type(pets):
    for pet in pets:
        if pet.get("animal_type") == "cat":
            print(f"animalType: {pet['animal_type']}")


def print_names_by_type(pets, animal_type):
    found = False
    for pet in pets:
        if pet.get("animal_type") == animal_type:
            print(f"Names of {animal_type}: {pet['names']}")
            found = True
    if not found:
        print(f"Animal type '{animal_type}' not found.")


def add_name_to_all(pets, name):
    for pet in pets:
        if name not in pet["names"]:
            pet["names"].append(name)
    return pets


if __name__ == "__main__":
    print("--- Print Animal Type 'cat' ---")
    print_cat_type(our_pets)

    print("\n--- Print Names for 'dog' ---")
    print_names_by_type(our_pets, "dog")

    print("\n--- Add 'Buddy' to all animal types ---")
    add_name_to_all(our_pets, "Buddy")
    print("Updated pets array:", our_pets)