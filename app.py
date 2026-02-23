from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "nexora-secret-key-2024")

# Setup Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Setup Gemini client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ===================================
# COMPANY INFORMATION — EDIT THIS!
# ===================================
SYSTEM_PROMPT = """
You are a friendly and professional assistant for Nexora Digital Solutions.
You help both employees and customers.

=== COMPANY INFO ===
Company Name: Nexora Digital Solutions Sdn. Bhd.
Mission: To provide quality technology solutions to businesses in Malaysia.
Vision: To be the top IT company in Southeast Asia by 2030.
CEO: (your CEO name)
Founded: (your founded year)
Location: (your office address)
Website: (your website)

=== FOR CUSTOMERS ===
Products/Services: (your products/services)
Support Email: support@nexoradigital.com
Support Phone: (your phone)
Working Hours: Monday-Friday, 9AM-6PM (MYT)

=== FOR EMPLOYEES ===
HR Contact: (HR email)
Leave Policy: (leave policy details)
Company Values: (company values)
IT Support: (IT support contact)

=== RULES ===
- Be friendly and professional at all times
- Only answer questions related to the company
- If you don't know the answer say "Please contact support@nexoradigital.com"
- Do not make up any information
- Keep answers short and clear
- Answer in the same language the user writes in
- Never use asterisks or markdown formatting in replies
"""

# Store chat history per session
chat_histories = {}

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = os.urandom(8).hex()
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    session_id = session.get("session_id", "default")

    if not user_message:
        return jsonify({"reply": "Please type a message."})

    # Get or create history for this session
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]

    # Keep only last 10 messages to save tokens
    if len(history) > 10:
        history = history[-10:]
        chat_histories[session_id] = history

    # Add user message to history
    history.append({
        "role": "user",
        "content": user_message
    })

    # Try Groq first, fallback to Gemini
    reply = try_groq(history)
    if reply is None:
        print("Groq failed, switching to Gemini...")
        reply = try_gemini(history)
    if reply is None:
        reply = "Sorry, I am currently unavailable. Please contact support@nexoradigital.com"

    # Add reply to history
    history.append({
        "role": "assistant",
        "content": reply
    })

    return jsonify({"reply": reply})


def try_groq(history):
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq error: {e}")
        return None


def try_gemini(history):
    try:
        # Convert history format for Gemini
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])]
                )
            )

        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=500
            ),
            contents=gemini_history
        )
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


@app.route("/reset", methods=["POST"])
def reset():
    session_id = session.get("session_id", "default")
    if session_id in chat_histories:
        chat_histories[session_id] = []
    return jsonify({"status": "Chat cleared!"})


if __name__ == "__main__":
    app.run(debug=True)