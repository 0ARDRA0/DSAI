import json
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
import pymongo
import jwt
import datetime
from fuzzywuzzy import process
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Load environment variables
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = SECRET_KEY
Session(app)

# MongoDB setup
client = pymongo.MongoClient(MONGODB_URI)
db = client["bia_chatbot"]
chat_collection = db["chats"]
user_collection = db["users"]

# Load your custom-trained LLaMA model
model_name = "trained_llama_model"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dummy in-memory database for RAG
# Load intents from JSON file
with open("data/intents.json", "r") as file:
    intents = json.load(file)

# Prepare knowledge base from intents
knowledge_base = []
for intent in intents["intents"]:
    for response in intent["responses"]:
        knowledge_base.append({"tag": intent["tag"], "content": response})


# Prepare TF-IDF for simple retrieval
vectorizer = TfidfVectorizer()
documents = [item["content"] for item in knowledge_base]
tfidf_matrix = vectorizer.fit_transform(documents)


def retrieve_documents(query):
    query_tfidf = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    best_match_idx = np.argmax(cosine_similarities)

    # Simple threshold-based retrieval
    if cosine_similarities[best_match_idx] > 0.1:  # Adjust threshold as needed
        return knowledge_base[best_match_idx]["content"]
    return None


def get_response(tag):
    for intent in intents:
        if intent["tag"] == tag:
            return intent["responses"]
    return [
        "I'm not sure I have the right answer to that question. Please try asking in a different way, or contact our support for further assistance."
    ]


def generate_llama_response(context):
    input_ids = tokenizer.encode(context, return_tensors="pt")
    output = model.generate(
        input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")

    user_input = request.json.get("message")

    # Retrieve relevant content (RAG process)
    retrieved_content = retrieve_documents(user_input)

    if retrieved_content:
        # Combine the retrieved content with user input
        augmented_input = f"{user_input}. Retrieved information: {retrieved_content}"
    else:
        augmented_input = user_input

    # Generate response with the LLaMA model
    inputs = tokenizer(augmented_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ip_address = request.remote_addr

    # Save chat history to MongoDB
    chat_collection.insert_one(
        {
            "ip": ip_address,
            "message": message,
            "response": response,
            "timestamp": datetime.datetime.utcnow(),
        }
    )
    return jsonify({"response": response})


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    user = user_collection.find_one({"username": username})

    if not user or not check_password_hash(user["password"], password):
        return jsonify({"message": "Invalid credentials!"}), 401

    token = jwt.encode(
        {
            "username": username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
        },
        SECRET_KEY,
        algorithm="HS256",
    )

    return jsonify({"token": token}), 200


@app.route("/history", methods=["GET"])
def get_history():
    token = request.headers.get("Authorization")

    if not token:
        return jsonify({"message": "Token is missing!"}), 401

    try:
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return jsonify({"message": "Token has expired!"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"message": "Invalid token!"}), 401

    # Fetch chat history from MongoDB
    chats = list(chat_collection.find({}, {"_id": 0}).sort("_id", pymongo.DESCENDING))

    return jsonify({"history": chats}), 200


if __name__ == "__main__":
    app.run(debug=True)
