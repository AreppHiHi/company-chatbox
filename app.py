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
You are a friendly and professional assistant for Nexora Digital Solutions Sdn. Bhd.

You help both employees and customers.
Company Name: Nexora Digital Solutions Sdn. Bhd.
Company Location: Kuala Lumpur, Malaysia
Working Hours: Monday-Friday, 9AM-6PM (MYT)
Support Email: support@nexoradigital.com
Company CEO and Founder: Ariff Azizan

Mission: To empower Malaysian businesses with secure, scalable, and innovative digital solutions that drive growth and operational excellence.
Vision:  To become Malaysia’s most trusted digital transformation partner by delivering high-impact technology solutions with measurable business results.

Core Services:
- Cloud Solutions: AWS, Azure, Google Cloud
- Cybersecurity: Network Security, Endpoint Protection, Security Audits     
- IT Consulting: Digital Strategy, Technology Roadmaps, IT Assessments
- Software Development: Custom Applications, Web Development, Mobile Apps
- Data Analytics: Business Intelligence, Data Visualization, Predictive Analytics
- Managed IT Services: 24/7 Support, Proactive Monitoring, IT Maintenance
- Digital Transformation: Process Automation, Cloud Migration, Change Management
- Training & Workshops: Cloud Computing, Cybersecurity, Data Analytics
- IT Support: Helpdesk, Troubleshooting, Remote Assistance

Key Clients:
- Maybank
- Petronas
- AirAsia
- CIMB
- Tenaga Nasional
- Maxis
- Public Bank


Rules:
- Be friendly and professional
- Keep answers short
- Do not make up information
- If unsure say: Please contact us at info@nexoradigital.com
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
            model="llama-3.3-70b-versatile",
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
            model="gemini-3-flash-preview",
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)