import os
import re
import random
from dataclasses import dataclass
from typing import List

import spacy
from spacy.symbols import ORTH

# Ensure Faker is installed: pip install Faker


# Utility function to load addresses from a file
def load_addresses(file_path):
    with open(file_path, "r") as file:
        addresses = [line.strip() for line in file.readlines()]
    return addresses


# Data class to store entity information
@dataclass
class TrainableEntity:
    tag: str
    replacement_function: callable


# Data class to handle the training process
@dataclass
class TrainingRun:
    model_name: str
    custom_entities: List[TrainableEntity]
    addresses: List[str]
    training_seed: int = 42
    base_model: str = "en_core_web_lg"
    batch_size: int = 2
    dropout_rate: float = 0.5

    def __post_init__(self):
        self.number_of_training_examples = len(self.addresses)
        self._replacement_func_dict = {
            ent.tag: ent.replacement_function for ent in self.custom_entities
        }
        entity_tags = "|".join([re.escape(ent.tag) for ent in self.custom_entities])
        self._replacement_pattern = r"\{(" + entity_tags + r")\}"

    @staticmethod
    def prepare_nlp_model(nlp):
        special_cases = [":", ";", "a/k/a", "f/k/a"]
        for case in special_cases:
            nlp.tokenizer.add_special_case(case, [{ORTH: case}])

        if "ner" not in nlp.pipe_names:
            nlp.add_pipe("ner", last=True)

    def run(self):
        print("Run for", self.model_name, "started")
        nlp = spacy.load(self.base_model)
        self.prepare_nlp_model(nlp)

        ner = nlp.get_pipe("ner")
        for ent in self.custom_entities:
            print("Adding entity label", ent.tag)
            ner.add_label(ent.tag)

        random.seed(self.training_seed)

        print("Generating training data...")

        training_data = []
        for address in self.addresses:
            training_data.extend(self.generate_annotated_documents(address))

        print("Generated training data.")

        print("Converting training data to spaCy's format...")
        examples = []
        from spacy.training import Example

        for text, annots in training_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annots)
            examples.append(example)

        print("Beginning training...")
        optimizer = nlp.resume_training()

        for itn in range(self.number_of_training_examples):
            random.shuffle(examples)
            losses = {}
            for batch in spacy.util.minibatch(examples, size=self.batch_size):
                nlp.update(batch, drop=self.dropout_rate, losses=losses, sgd=optimizer)
            print(f"Iteration {itn}, Losses {losses}")

        print("Saving model...")
        model_path = os.path.join("../models", self.model_name)
        os.makedirs(model_path, exist_ok=True)
        nlp.to_disk(model_path)

    def generate_annotated_documents(self, address) -> List[tuple[str, dict]]:
        templates = [
            "BEING KNOWN AS {ADDRESS}",
            "Premises Being: {ADDRESS}",
            "PROPERTY ADDRESS: {ADDRESS}",
            "Being known as {ADDRESS}",
            "Property Address: {ADDRESS}",
            "Located at {ADDRESS}",
            "Address: {ADDRESS}",
            "At {ADDRESS}",
        ]
        documents = []
        for template in templates:
            filled_template = template
            offset = 0
            entities = []

            matches = list(re.finditer(self._replacement_pattern, filled_template))

            for match in matches:
                text_type = match.group(1)
                start, end = match.start() + offset, match.end() + offset
                filled_template = (
                    filled_template[:start] + address + filled_template[end:]
                )
                offset += len(address) - (end - start)
                entities.append((start, start + len(address), text_type))

            documents.append((filled_template, {"entities": entities}))

        return documents


# Create training data with address examples
def train(addresses):
    trainer = TrainingRun(
        model_name="address_ner_model",
        custom_entities=[
            TrainableEntity(
                "ADDRESS", replacement_function=lambda: random.choice(addresses)
            )
        ],
        addresses=addresses,
    )
    trainer.run()


if __name__ == "__main__":
    address_file_path = (
        "../data/training_address_data.txt"  # Update with the path to your address file
    )
    addresses = load_addresses(address_file_path)
    print(addresses)
    train(addresses)
