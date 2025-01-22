# Required Libraries
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the Summarization Pipeline
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Function to Summarize Text
def summarize_text(text, max_length=1000, min_length=50):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize')
def summarize():
    return render_template('summarize.html')

@app.route('/process', methods=['POST'])
def process_text():
    # Get user input from the form
    input_text = request.form['input_text']
    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400

    # Process the text
    summary = summarize_text(input_text)

    # Return results
    return jsonify({'summary': summary})

if __name__ == "__main__":
    app.run(debug=True)
