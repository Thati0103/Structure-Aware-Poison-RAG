from pathlib import Path

import database
import prepare_data as prepare


def ask_poisoned():
    while True:
        answer = input("Poisoned document(s)? (y/n): ").strip().lower()

        if answer in ("y", "n"):
            return answer == "y"

        print("Please enter y or n.")


def ask_pdf():
    filepath = Path(input("PDF path: ").strip()).expanduser()

    if not filepath.is_file() or filepath.suffix.lower() != ".pdf":
        raise ValueError("Please provide an existing PDF file.")

    return str(filepath), filepath.name


def main():
    while True:
        print("""
            1. Process all PDFs in a folder
            2. Process one PDF
            3. Delete one document from the database
            4. Delete folder documents from the database
            5. Create all tables
            6. Drop all tables
            7. Exit
            """)
        choice = input("Choose an option: ").strip()

        try:
            if choice == "1":
                prepare.prepare_data(ask_poisoned())

            elif choice == "2":
                filepath, filename = ask_pdf()
                status, document_id = prepare.insert_document(
                    filepath, filename, ask_poisoned()
                )
                if status:
                    print(f"Saved document {document_id}.")

            elif choice == "3":
                filepath = input("Stored PDF path: ").strip()
                prepare.delete_document(filepath, Path(filepath).name)

            elif choice == "4":
                prepare.delete_data(ask_poisoned())

            elif choice == "5":
                database.create_schema()

            elif choice == "6":
                database.drop_database()

            elif choice == "7":
                break

            else:
                print("Please choose 1–7.")

        except Exception as error:
            print(f"Operation failed: {error}")


if __name__ == "__main__":
    main()