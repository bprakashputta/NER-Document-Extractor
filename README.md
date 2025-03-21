# NER Document Extractor

This project provides a comprehensive pipeline to extract and evaluate addresses from legal PDF documents using Named Entity Recognition (NER) techniques. Built with Python, spaCy, and pdfminer, the pipeline includes data preprocessing, model training, address extraction, and performance evaluation.

## Project Structure
```
.
├── README.md
├── data
├── final_output
├── metrics_output
├── models
├── pdf_text_extraction_results
└── scripts
```

## Key Functionalities

### 1. PDF Text Extraction
- Uses `pdfminer` to convert PDF documents into plain text.
- Includes basic text cleaning and analysis to assist with data preprocessing.

**Scripts:**
- `scripts/processor_util.py`

### 2. Training NER Model
- Custom-trained spaCy model for address recognition.
- Generates synthetic training data from address templates.
- Implements data augmentation and randomized batch training.

**Scripts:**
- `scripts/train_ner.py`

### 3. Address Extraction
- Leverages the trained spaCy NER model to extract addresses from text files.
- Cleans and validates extracted addresses using regular expressions.
- Stores results in structured formats (CSV, JSON, TXT).

**Scripts:**
- `scripts/test_ner.py`

### 4. Performance Evaluation
- Calculates precision, recall, and F1-score based on extracted addresses.
- Outputs detailed metrics, including true positives, false positives, and false negatives.
- Stores evaluation metrics in CSV and JSON for analysis.

**Scripts:**
- `scripts/performance_util.py`

## Usage

### Step-by-step workflow:
1. **Extract text from PDFs**:
```bash
python scripts/processor_util.py
```

2. **Train NER model**:
```bash
python scripts/train_ner.py
```

3. **Extract and validate addresses**:
```bash
python scripts/test_ner.py
```

4. **Evaluate model performance**:
```bash
python scripts/performance_util.py
```

## Dependencies
- spaCy
- pdfminer.six
- Faker (for data augmentation)

Install dependencies via:
```bash
pip install spacy pdfminer.six Faker
python -m spacy download en_core_web_lg
```

## Output
- Extracted addresses: `final_output/`
- Model performance metrics: `metrics_output/`
- Trained spaCy model: `models/address_ner_model`

## Contributions
Feel free to open issues, suggest improvements, or submit pull requests to enhance the project.

## License
This project is open-source and available under the MIT License.

# NER-Document-Extractor
