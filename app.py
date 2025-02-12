from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import spacy
from typing import List, Dict
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import os


from processing import generate_mnemonic, generate_story, generate_summary




app = Flask(__name__)
CORS(app) 
CORS(app, origins=["https://bodhiment.vercel.app"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_text(sample_text):
    # Generate summaries and mnemonic
    mnemonic = generate_mnemonic(sample_text)
    summary = generate_summary(sample_text)
    story = generate_story(sample_text)

    # Return a JSON response
    response_json = {
        "story": story,
        "key_points": "na ",
        "main_topics": "na ",
        "mnemonic": mnemonic,
        "summary": summary
    }

    return response_json



@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    print(data)
    input_text = data.get('input_text', '')
    selected_model = data.get('model', '')  
    processed_html = process_text(input_text)
    response = jsonify({'generated_sentence': processed_html})
    response.headers.add("Access-Control-Allow-Origin", "https://bodhiment.vercel.app")
    return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
