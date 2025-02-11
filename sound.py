import requests
import nltk
import pyttsx3
from nltk.corpus import words
from nltk.tokenize import SyllableTokenizer

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("words")

# Initialize components
engine = pyttsx3.init()  # Text-to-Speech Engine
tokenizer = SyllableTokenizer()
word_list = set(words.words())

# Rule-based predefined mnemonics
rule_based_mnemonics = {
    "chromosome": "chrome of home",
    "amplifier": "amp life ear",
    "photosynthesis": "photo sin thesis",
    "metamorphosis": "meta morph osis",
    "thermometer": "thermo meter",
    "encyclopedia": "encyclo pedia",
    "catastrophe": "cat as trophy",
    "archaeology": "arch ae o logy",
    "parallelogram": "parallel o gram",
    "pneumonia": "new moan ia",
    "hemisphere": "he miss sphere",
    "biology": "bio logy",
    "psychology": "psycho logy",
    "restaurant": "rest aura ant",
    "exaggerate": "egg agger ate",
    "millennium": "mill ennium",
    "philosophy": "phi loso phy",
    "temperature": "temp era ture",
    "vocabulary": "voca bul ary",
    "hypothesis": "hi pot he sis",
    "photosynthesis": "photo sin thesis",
    "electrolysis": "electro lysis",
    "oxidation": "ox id ation",
    "respiration": "res piration",
    "chromosome": "chrome of home",
    "refraction": "refract ion",
    "catalyst": "cat a lyst",
    "nucleotide": "nucleo tide",
    "transpiration": "trans piration",
    "amphibian": "am phi bian",
    "evaporation": "eva pore action",
    "neutralization": "new true lie station",
    "diffusion": "diff use ion",
    "fertilization": "fertile ice action",
    "valency": "value NC",
    "herbivore": "herb I vore",
    "isotopes": "iso topes",
    "centrifuge": "centre if huge",
    "hydrolysis": "hydro lysis",
    "malleability": "mallet ability",
    "respiration": "rest pure action",
    "photosynthesis": "photo sin thesis",
    "sublimation": "sub lime action",
    "electrolysis": "electric low lysis",
    "decomposition": "decompose position",
    "condensation": "cone dense station",
    "oxidation": "oxy date ion",
    "reflection": "re flex action",
    "digestion": "dig guest ion",
    "conduction": "cone duck tion",
    "deadlock": "dead lock",
    "chromosome": "chrome some",
    "amplifier": "amp life ear",
    "elephant": "ele phant",
    "butterfly": "butter fly",
    "keyboard": "key board",
    "firewall": "fire wall",
    "password": "pass word",
    "hardware": "hard ware",
    "software": "soft ware",
}
def speak_text(text):
    """Convert text to speech with replay option."""
    while True:
        engine.say(text)
        engine.runAndWait()
        
        replay = input("\n🔁 Do you want to hear it again? (Y/N): ").strip().lower()
        if replay not in {'y', 'yes'}:
            break

def get_meaningful_word(part):
    """Fetch a meaningful phonetic alternative using Datamuse API."""
    url = f"https://api.datamuse.com/words?sl={part}&max=5"
    try:
        response = requests.get(url, timeout=5)  # Add timeout
        response.raise_for_status()  # Raise error for bad response
        data = response.json()

        for item in data:
            word = item["word"]
            if word in word_list and len(word) >= 3:  # Ensure it's a real, meaningful word
                return word

    except requests.exceptions.RequestException:
        pass
    return part  # If no alternative, return original part

def split_word(word):
    """Split the word into 2-4 meaningful parts dynamically, ensuring all parts have meaning."""
    
    # First, check if the word has a predefined mnemonic
    if word.lower() in rule_based_mnemonics:
        return rule_based_mnemonics[word.lower()]
    
    # If no rule-based mnemonic, proceed with Datamuse method
    syllables = tokenizer.tokenize(word)

    if len(syllables) < 2:
        return word  # If word is too short, return as is

    # Try splitting into 2 parts
    mid = len(syllables) // 2
    first_part, second_part = "".join(syllables[:mid]), "".join(syllables[mid:])

    if first_part in word_list and second_part in word_list:
        return f"{first_part} {second_part}"  # Direct 2-word split

    # Try splitting into 3-4 parts
    for split_count in range(3, 5):
        step = max(1, len(syllables) // split_count)
        parts = ["".join(syllables[i:i + step]) for i in range(0, len(syllables), step)]

        refined_parts = [get_meaningful_word(part) for part in parts]
        return " ".join(refined_parts)

    return word  # Return original word if no split works



def generate_mnemonic_for_word(user_word):
    if user_word.lower() == "exit":
        print("👋 Exiting... Goodbye!")
        return  # exit the function if the word is 'exit'

    mnemonic = split_word(user_word)  # Assuming split_word is a function you have defined
    response_json = {
        "story": "",
        "key_points": "",
        "main_topics": "",
        "mnemonic": mnemonic,
        "summary": ""
    }
    return response_json
    print(f"📌 Mnemonic for '{user_word}': {mnemonic}")
    speak_text(f"Mnemonic for {user_word} is {mnemonic}")  # Assuming speak_text is a function you have defined

# Example of calling the function
