# Required Libraries
from transformers import pipeline
import spacy

# Load the Summarization Pipeline
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Load SpaCy Model for Key Concept Extraction
nlp = spacy.load("en_core_web_sm")

# Function to Summarize Text
def summarize_text(text, max_length=10000000, min_length=100):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Function to Extract Key Concepts
def extract_key_concepts(text):
    doc = nlp(text)
    concepts = set()
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ in ("nsubj", "dobj", "pobj"):
            concepts.add(chunk.text.strip())
    return list(concepts)

# Main Program
def study_genie():
    print("Welcome to StudyGenie: Your Learning Assistant!")
    print("Enter your text (or type 'exit' to quit):\n")
    
    while True:
        user_input = input("\nEnter Text: ")
        if user_input.lower() == 'exit':
            print("Thank you for using StudyGenie!")
            break

        print("\nProcessing your input...\n")

        # Summarize the Text
        summary = summarize_text(user_input)
        print(f"Summary:\n{summary}\n")

        # Extract Key Concepts
        key_concepts = extract_key_concepts(user_input)
        print("Key Concepts:")
        for i, concept in enumerate(key_concepts, 1):
            print(f"{i}. {concept}")

# Run StudyGenie
if __name__ == "__main__":
    study_genie()
