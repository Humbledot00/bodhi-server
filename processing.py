import spacy
from typing import List, Dict
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    
########################################################################################
# TEXT ANALYSIS (Key Points Extraction)
########################################################################################
class TextAnalyzer:
    def __init__(self):
        # Load English language model
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        text = re.sub(r'[^\w\s.]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        return text.strip()

    def get_key_points(self, text: str) -> Dict[str, List[str]]:
        """Extract key points from the input text using NLP techniques."""
        cleaned_text = self.preprocess_text(text)
        doc = self.nlp(cleaned_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Calculate sentence importance using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Calculate sentence scores
        sentence_scores = []
        for i, sent in enumerate(doc.sents):
            score = np.sum(tfidf_matrix[i].toarray())
            entities_bonus = len([ent for ent in sent.ents])
            pos_bonus = len([token for token in sent if token.pos_ in ['NOUN', 'VERB', 'PROPN']])
            total_score = score + (0.1 * entities_bonus) + (0.05 * pos_bonus)
            sentence_scores.append((sent.text.strip(), total_score))
        
        ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        entities = [ent.text for ent in doc.ents]
        key_entities = Counter(entities).most_common(5)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        main_topics = Counter(noun_chunks).most_common(5)
        
        return {
            'key_points': [sent for sent, score in ranked_sentences[:3]],
            'main_topics': [topic for topic, count in main_topics],
        }

########################################################################################
# MNEMONIC GENERATION
########################################################################################
def generate_mnemonic(sample_text):
    """Generate a creative mnemonic for given text and remove the prompt from the output."""
    
    prompt = f"Create a creative and simple mnemonic to remember this content easily:\n\n{sample_text}\n\nMnemonic:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Remove any part of the prompt from the output
    clean_output = generated_text.replace(prompt, "").strip()

    return clean_output

########################################################################################
# STORY GENERATION
########################################################################################
def generate_story(sample_text):
    """Generate a creative short story based on a given idea."""
    torch.set_default_device("cuda")
    prompt = f"Write a simple and creative story based on the following idea: {sample_text}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    if "##OUTPUT:" in generated_text:
        generated_text = generated_text.split("##OUTPUT:")[-1].strip()
    clean_output = generated_text.replace(prompt, "").strip()

    return clean_output

########################################################################################
# Summary GENERATION
########################################################################################
def generate_summary(sample_text):
    # Ensure CUDA is available before setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = (
        "Summarize the following text in a simple way to help remember it easily:\n\n"
        f"{sample_text}\n\n"
        "## Summary:\n"
    )
    # Tokenize and move input to the correct device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate summary
    outputs = model.generate(**inputs, max_length=250, pad_token_id=tokenizer.eos_token_id)
    # Decode output
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Extract summary more reliably
    if "## Summary:" in generated_text:
        generated_text = generated_text.split("## Summary:")[-1].strip()
    clean_output = generated_text.replace(prompt, "").strip()

    return clean_output

