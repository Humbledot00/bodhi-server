from flask import Flask, request, jsonify
from flask_cors import CORS
import random

from processing import TextAnalyzer, TextSimplifier, generate_mnemonic, generate_story, generate_summary


app = Flask(__name__)
CORS(app) 


def process_text(sample_text):
    analyzer = TextAnalyzer()
    simplifier = TextSimplifier()

    # Extract key points and topics
    result = analyzer.get_key_points(sample_text)
    key_points = result['key_points']
    main_topics = result['main_topics']

    # Generate summaries and mnemonic
    simplified_text = simplifier.simplify_text(sample_text)
    mnemonic = generate_mnemonic(sample_text)
    summary = generate_summary(sample_text)
    story = generate_story(sample_text)

    # Return a JSON response
    response_json = {
        "story": story,
        "key_points": key_points,
        "main_topics": main_topics,
        "simplified_text": simplified_text,
        "mnemonic": mnemonic,
        "summary": summary
    }

    return response_json



@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    print(data)
    input_text = data.get('input_text', '')
    processed_html = "process_text(input_text)"
    return jsonify({'generated_sentence': processed_html})


if __name__ == '__main__':
    app.run(debug=True)