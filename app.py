from flask import Flask, render_template, request, jsonify
import nltk
import random
import string
import os
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# Define intents and keywords
intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "thanks": ["thanks", "thank you", "thx"],
    "courses": ["Availability of courses"],
    "fee_details": ["Fees structure"],
    "cs": ["BSC(CS)"],
    "bsc_ds": ["BSC(DS)"],
    "ai": ["BSC(AI)"],
    "mbc": ["BSC(MBC)"],
    "bca": ["BCA"],
    "bca_ds": ["BCA(DS)"],
    "bba_dm": ["BBA(DM)"],
    "bba_gn": ["BBA(GEN)"],
    "placements": ["What about placements"],

    # New intents
    "placements_2022_2023": [
        "placements 2022", "placements 2023", "2022 placements", "2023 placements",
        "placements 2022-2023", "placement history", "placement report"
    ],
    "placement_image_2025": ["2025 placement image", "placement image 2025", "show 2025 placement", "image of placement 2025"],
    "admission_count_2022_2023": ["admission count 2022-2023", "admission 2022", "2022 admission count"],
    "admission_count_2023_2024": ["admission count 2023-2024", "admission 2023", "2023 admission count"],
    "admission_count_2024_2025": ["admission count 2024-2025", "admission 2024", "2024 admission count"],

    "bus": ["Bus availability"],
    "hostel": ["Hostel facilities"],
    "contact_support": ["Contact details"],
    "address": ["Address"],
    "default": []
}

# Define responses
responses = {
    "greeting": ["Hello! This is Aditya Degree College for Women, Kakinada. How can I assist you today?"],
    "thanks": ["You're welcome!"],
    "courses": ["BSC with specializations of DS, AI, CS, BCA(DS), BCA, BBA(GEN), BBA(DM), MBC"],
    "fee_details": ["""
    <table border="1" style="border-collapse: collapse; width:100%; text-align:center;">
    <tr><th>Course</th><th>2025-26 (1st Year)</th><th>2026-27 (2nd Year)</th><th>2027-28 (3rd Year)</th></tr>
    <tr><td>B.Sc (Computers)</td><td>55000</td><td>56000</td><td>57000</td></tr>
    <tr><td>B.Sc (Data Science)</td><td>58000</td><td>59000</td><td>60000</td></tr>
    <tr><td>B.Sc (Artificial Intelligence)</td><td>58000</td><td>59000</td><td>60000</td></tr>
    <tr><td>B.Sc (Microbiology)</td><td>40000</td><td>41000</td><td>42000</td></tr>
    <tr><td>BCA</td><td>62000</td><td>63000</td><td>64000</td></tr>
    <tr><td>BCA (Data Science)</td><td>62000</td><td>63000</td><td>64000</td></tr>
    <tr><td>BBA - DM</td><td>65000</td><td>66000</td><td>67000</td></tr>
    <tr><td>BBA - General</td><td>60000</td><td>61000</td><td>62000</td></tr>
    </table>
    """],
    "cs": ["1st-55K, 2nd-56K, 3rd-57K"],
    "bsc_ds": ["1st-58k, 2nd-59k, 3rd-60k"],
    "ai": ["1st-58k, 2nd-59k, 3rd-60k"],
    "mbc": ["1st-40k, 2nd-41k, 3rd-42k"],
    "bca": ["1st-62k, 2nd-63k, 3rd-64k"],
    "bca_ds": ["1st-62k, 2nd-63k, 3rd-64k"],
    "bba_dm": ["1st-65k, 2nd-66k, 3rd-67k"],
    "bba_gn": ["1st-60k, 2nd-61k, 3rd-62k"],
    "placements": ["1600+ placements in 2025. Highest package is 7-8 LPA."],

    # New responses
    "placements_2022_2023": [
        "Please visit: <a href='https://aditya.ac.in/awdckkd/placements.php?campus=2' target='_blank'>Placement Report 2022–2023</a>"
    ],
    "placement_image_2025": [
        "<img src='https://i.postimg.cc/BQsdLzD6/1.jpg' alt='2025 Placement' width='100%'/>"
    ],
    "admission_count_2022_2023": ["Total admission count: 586"],
    "admission_count_2023_2024": ["Total admission count: 656"],
    "admission_count_2024_2025": ["Total admission count: 573"],

    "bus": ["Yes, we have bus facility. Fees depend on the distance."],
    "hostel": ["Yes, we offer safe & secure hostel with hygienic food and Wi-Fi. ₹90,000/year."],
    "contact_support": ["Contact: 7036966663, Email: awdckkd@gmail.com"],
    "address": ["Aditya Degree College For Women, Sambamurthy Nagar, Kakinada"],
    "default": ["I'm sorry, I didn't understand that. Could you please rephrase?"]
}

# Preprocessing
def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    return [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]

# Intent prediction
def predict_intent(user_input):
    processed_input = preprocess(user_input)
    best_intent = "default"
    max_matches = 0

    for intent, keywords in intents.items():
        keyword_tokens = preprocess(" ".join(keywords))
        matches = sum(1 for word in processed_input if word in keyword_tokens)
        if matches > max_matches:
            max_matches = matches
            best_intent = intent

    return best_intent

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["msg"]
    intent = predict_intent(user_input)
    response = random.choice(responses[intent])
    return jsonify({"response": response})

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
