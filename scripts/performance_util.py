import csv
import json
import spacy
import os
import glob
import re

# Load the NER model once globally
nlp = spacy.load("../models/address_ner_model")

# Function to extract addresses from text using the NER model
def extract_addresses(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ADDRESS"]

# Function to clean addresses
def clean_address(address):
    address = re.sub(r"\s+", " ", address).strip()
    address = re.sub(r"[.,]+$", "", address)
    address = re.sub(
        r"\b(?:Estate|Executor|Executrix|Administrator|Administratrix)\b.*$",
        "",
        address,
    )
    address = re.sub(r"[^\w\s,]", "", address)
    return address

# Function to validate if a string is a likely address
def is_likely_address(text):
    return bool(
        re.search(
            r"\d+\s[A-Za-z]+(?:\s[A-Za-z]+)+,?\s+[A-Za-z]+(?:\s[A-Za-z]+)*,\s+[A-Za-z]+(?:\s[A-Za-z]+)*,?\s+\d{5}",
            text,
        )
    )

# Function to calculate performance metrics
def calculate_performance_metrics(predicted_addresses, actual_addresses):
    predicted_set = set(predicted_addresses)
    actual_set = set(actual_addresses)

    true_positives = predicted_set.intersection(actual_set)
    false_positives = predicted_set - actual_set
    false_negatives = actual_set - predicted_set

    true_positives_count = len(true_positives)
    false_positives_count = len(false_positives)
    false_negatives_count = len(false_negatives)
    true_negatives_count = 0  # No ground truth data for true negatives

    precision = (
        true_positives_count / (true_positives_count + false_positives_count)
        if (true_positives_count + false_positives_count) > 0
        else 0
    )
    recall = (
        true_positives_count / (true_positives_count + false_negatives_count)
        if (true_positives_count + false_negatives_count) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "true_positives_count": true_positives_count,
        "false_positives_count": false_positives_count,
        "false_negatives_count": false_negatives_count,
        "true_negatives_count": true_negatives_count,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": list(true_positives),
        "false_positives": list(false_positives),
        "false_negatives": list(false_negatives)
    }

# Function to write performance metrics to a CSV file
def write_performance_metrics_to_csv(metrics, output_file):
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "metric",
            "value"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write metrics to CSV
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):  # Write numerical metrics
                writer.writerow({"metric": metric, "value": value})
            elif isinstance(value, list):  # Write lists as separate rows
                for item in value:
                    writer.writerow({"metric": f"{metric}_value", "value": item})

# Function to write performance metrics to a JSON file
def write_performance_metrics_to_json(metrics, output_file):
    with open(output_file, "w") as jsonfile:
        json.dump(metrics, jsonfile, indent=4)

# Function to process each file in the directory
def process_files_in_directory(input_dir, output_dir):
    # Ensure the output directories exist for metrics
    metrics_output_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_output_dir, exist_ok=True)

    # Initialize accumulators for all metrics
    all_predicted_addresses = []
    all_actual_addresses = []

    # Iterate through all text files in the input directory
    input_files = glob.glob(os.path.join(input_dir, "*.txt"))
    for file_path in input_files:
        # Read the text from each file
        with open(file_path, "r") as file:
            text = file.read()

        # Extract addresses using the NER model
        predicted_addresses = extract_addresses(text)

        # Clean and filter addresses
        cleaned_addresses = [
            clean_address(addr)
            for addr in predicted_addresses
            if is_likely_address(addr)
        ]
        cleaned_addresses = list(set(cleaned_addresses))  # Remove duplicates

        # Accumulate addresses
        all_predicted_addresses.extend(predicted_addresses)
        all_actual_addresses.extend(cleaned_addresses)

        print(f"Processed: {file_path}")

    # Calculate overall performance metrics
    overall_metrics = calculate_performance_metrics(
        predicted_addresses=all_predicted_addresses,
        actual_addresses=all_actual_addresses
    )

    # Write overall performance metrics to CSV
    metrics_output_file_csv = os.path.join(metrics_output_dir, "overall_metrics.csv")
    write_performance_metrics_to_csv(overall_metrics, metrics_output_file_csv)

    # Write overall performance metrics to JSON
    metrics_output_file_json = os.path.join(metrics_output_dir, "overall_metrics.json")
    write_performance_metrics_to_json(overall_metrics, metrics_output_file_json)

if __name__ == "__main__":
    input_dir = "../pdf_text_extraction_results"
    output_dir = "../metrics_output"

    process_files_in_directory(input_dir, output_dir)
