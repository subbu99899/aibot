from flask import Flask, render_template, request, jsonify
import nltk
import random
import string
from nltk.stem import WordNetLemmatizer
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "thanks": ["thanks", "thank you", "thx"],
    "courses": ["Availability of courses"],
    "fee_details": ["Fees structure"],
    "cs": ["BSC(CS)"],
    "ds": ["BSC(DS)"],
    "ai": ["BSC(AI)"],
    "mbc": ["BSC(MBC)"],
    "bca": ["BCA"],
    "bca_ds": ["BCA(DS)"],
    "bba_dm": ["BBA(DM)"],
    "bba_gn": ["BBA(GEN)"],
    "placements": ["What about placements"],
    "bus": ["Bus availability"],
    "hostel": ["Hostel facilities"],
    "contact_support": ["Contact details"],
    "address": ["Address"],
    "default": ["I'm sorry, I didn't understand that. Could you please rephrase?"]
}

responses = {
    "greeting": ["Hello! This is Aditya Degree College for Women's, How can I assist you today?"],
    "thanks": ["You're welcome!"],
    "courses": ["BSC with specializations of DS, AI, CS, BCA(DS), BCA, BBA(GEN), BBA(DM), MBC"],
    "fee_details": ["It's based on the group you had selected, concession may apply based on your Intermediate %"],
    "cs": ["1st-55K, 2nd-56K, 3rd-57K"],
    "ds": ["1st-58k, 2nd-59k, 3rd-60k"],
    "ai": ["1st-58k, 2nd-59k, 3rd-60k"],
    "mbc": ["1st-40k, 2nd-41k, 3rd-42k"],
    "bca": ["1st-62k, 2nd-63k, 3rd-64k"],
    "bca_ds": ["1st-62k, 2nd-63k, 3rd-64k"],
    "bba_dm": ["1st-65k, 2nd-66k, 3rd-67k"],
    "bba_gn": ["1st-60k, 2nd-61k, 3rd-62k"],
    "placements": ["1600+ placements in 2025. Highest package is 12 LPA."],
    "bus": ["Yes, we have bus facility. Fees depend on the distance."],
    "hostel": ["Yes, we offer safe & secure hostel with hygienic food and Wi-Fi. ₹90,000/year."],
    "contact_support": ["Contact: 7036966663, Email: awdckkd@gmail.com"],
    "address": ["Aditya Degree College For Women's, Sambamurthy Nagar, Kakinada"],
    "default": ["I'm sorry, I didn't understand that. Could you please rephrase?"]
}

def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    return [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]

def predict_intent(user_input):
    processed = preprocess(user_input)
    for intent, keywords in intents.items():
        if any(word in processed for word in preprocess(" ".join(keywords))):
            return intent
    return "default"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["msg"]
    intent = predict_intent(user_input)
    response = random.choice(responses[intent])
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
