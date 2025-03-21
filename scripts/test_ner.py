import spacy
import os
import glob
import re
import csv

# Load the NER model once globally
nlp = spacy.load("../models/address_ner_model")

# Function to extract addresses from text using the NER model
def extract_addresses(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ADDRESS"]

# Function to clean addresses
def clean_address(address):
    # Remove extra spaces and line breaks
    address = re.sub(r"\s+", " ", address).strip()
    # Remove any trailing commas or periods
    address = re.sub(r"[.,]+$", "", address)
    # Remove unnecessary details (adjust regex as needed)
    address = re.sub(
        r"\b(?:Estate|Executor|Executrix|Administrator|Administratrix)\b.*$",
        "",
        address,
    )
    # Remove unwanted characters
    address = re.sub(r"[^\w\s,]", "", address)
    return address

# Function to validate if a string is a likely address
def is_likely_address(text):
    # Improved heuristic for address validation
    return bool(
        re.search(
            r"\d+\s[A-Za-z]+(?:\s[A-Za-z]+)+,?\s+[A-Za-z]+(?:\s[A-Za-z]+)*,\s+[A-Za-z]+(?:\s[A-Za-z]+)*,?\s+\d{5}",
            text,
        )
    )

# Function to process each file in the directory and save results to a CSV file
def process_files_in_directory(input_dir, output_file):
    # List to store all results
    all_results = []

    # Iterate through all text files in the input directory
    input_files = glob.glob(os.path.join(input_dir, "*.txt"))
    for file_path in input_files:
        # Read the text from each file
        with open(file_path, "r") as file:
            text = file.read()

        # Extract addresses using the NER model
        extracted_addresses = extract_addresses(text)

        # Clean and filter addresses
        cleaned_addresses = [
            clean_address(addr)
            for addr in extracted_addresses
            if is_likely_address(addr)
        ]
        cleaned_addresses = list(set(cleaned_addresses))  # Remove duplicates

        # Add results to the list
        for addr in cleaned_addresses:
            all_results.append([os.path.basename(file_path), addr])

    # Write all results to a CSV file
    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Filename", "Address"])
        csvwriter.writerows(all_results)

    print(f"Processed {len(input_files)} files and saved results to {output_file}")

if __name__ == "__main__":
    input_dir = "../pdf_text_extraction_results"
    output_file = "../final_output/addresses.csv"

    process_files_in_directory(input_dir, output_file)
