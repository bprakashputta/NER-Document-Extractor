from pdfminer.high_level import extract_text
import os
import re


def extract_text_from_pdf(file_path):
    """Extract text from the given PDF file and return it."""
    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None


def save_text_to_file(text, output_path):
    """Save extracted text to a file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Error saving text to {output_path}: {e}")


def basic_analysis(text):
    """Perform basic analysis on the extracted text."""
    num_lines = len(text.split("\n"))
    num_words = len(text.split())
    print(f"Number of lines: {num_lines}")
    print(f"Number of words: {num_words}")


def clean_text(text):
    """Clean extracted text by removing extra whitespace."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main(input_folder, output_folder):
    """Main function to iterate over PDF files, extract text, save to files, and perform analysis."""
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".pdf"):
                pdf_file_path = os.path.join(root, file_name)
                output_file_name = os.path.splitext(file_name)[0] + ".txt"
                output_file_path = os.path.join(output_folder, output_file_name)

                # Extract text from PDF
                text = extract_text_from_pdf(pdf_file_path)
                if text:
                    print(f"Text extracted from {file_name} successfully.")

                    # Clean text (if needed)
                    cleaned_text = clean_text(text)

                    # Save extracted text to file
                    save_text_to_file(cleaned_text, output_file_path)
                    print(f"Text saved to {output_file_name}.")

                    # Perform basic analysis
                    basic_analysis(cleaned_text)
                    print("-------------------------")


if __name__ == "__main__":
    input_folder = "../data"
    output_folder = "../pdf_text_extraction_results"
    main(input_folder, output_folder)
