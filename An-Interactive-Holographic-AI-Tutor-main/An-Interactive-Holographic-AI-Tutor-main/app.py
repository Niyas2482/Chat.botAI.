
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    send_file,
)

from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)

from werkzeug.security import generate_password_hash, check_password_hash

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

from nltk.stem import WordNetLemmatizer
from gtts import gTTS

import numpy as np
import json
import pickle
import nltk
import random
import io
import os
import requests
import uuid


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

NLTK_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nltk_data"
)

nltk.data.path.insert(0, NLTK_DATA_DIR)

OUTPUT_FOLDER = "static/hologram"

app = Flask(__name__)

SECRET_KEY = os.environ.get("SECRET_KEY")

if not SECRET_KEY:
    if os.environ.get("APP_ENV") == "production":
        raise RuntimeError(
            "SECRET_KEY must be configured in production."
        )

    SECRET_KEY = "dev-only-secret-key-change-me"

app.config["SECRET_KEY"] = SECRET_KEY

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///edubot.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"


# --------------------------------------------------
# CHATBOT MODEL LOADING
# --------------------------------------------------

lemmatizer = WordNetLemmatizer()

MODEL_LOADED = False

try:
    from tensorflow.keras.models import load_model

    model = load_model("chatbot_model.h5")

    with open("words.pkl", "rb") as file:
        words = pickle.load(file)

    with open("classes.pkl", "rb") as file:
        classes = pickle.load(file)

    with open("intents.json", encoding="utf-8") as file:
        intents = json.load(file)

    MODEL_LOADED = True

    print("Chatbot model loaded successfully.")

except Exception as e:
    print(f"Model load error: {e}")
    MODEL_LOADED = False


# --------------------------------------------------
# DATABASE MODELS
# --------------------------------------------------

class User(UserMixin, db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    full_name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    username = db.Column(
        db.String(80),
        unique=True,
        nullable=False
    )

    password_hash = db.Column(
        db.String(256),
        nullable=False
    )

    grade = db.Column(db.String(20))

    school = db.Column(db.String(150))

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    messages = db.relationship(
        "ChatMessage",
        backref="user",
        lazy=True
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(
            self.password_hash,
            password
        )


class ChatMessage(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    sender = db.Column(
        db.String(10),
        nullable=False
    )

    message = db.Column(
        db.Text,
        nullable=False
    )

    timestamp = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    session_id = db.Column(
        db.String(50)
    )


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

with app.app_context():
    db.create_all()


# --------------------------------------------------
# LANGUAGE DETECTION AND TRANSLATION
# --------------------------------------------------

def detect_input_language(text, selected_language=None):

    script_ranges = {
        "ta": (0x0B80, 0x0BFF),
        "hi": (0x0900, 0x097F),
        "te": (0x0C00, 0x0C7F),
        "ml": (0x0D00, 0x0D7F),
        "kn": (0x0C80, 0x0CFF),
        "ar": (0x0600, 0x06FF),
    }

    for language, (start, end) in script_ranges.items():

        if any(
            start <= ord(char) <= end
            for char in text
        ):
            return language

    return "autodetect"


def translate_text(
    text,
    target_language="en",
    source_language=None
):

    if not text:
        return text

    source_language = source_language or "autodetect"

    if source_language == "auto":
        source_language = "autodetect"

    if source_language == target_language:
        return text

    url = "https://api.mymemory.translated.net/get"

    params = {
        "q": text,
        "langpair": f"{source_language}|{target_language}",
        "mt": "1",
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    result = response.json()

    if str(result.get("responseStatus")) != "200":

        details = str(
            result.get("responseDetails", "")
        )

        if (
            source_language == "autodetect"
            and target_language == "en"
            and "PLEASE SELECT TWO DISTINCT LANGUAGES"
            in details.upper()
        ):
            return text

        raise RuntimeError(
            f"Translation service failed: {details}"
        )

    translated_text = (
        result.get("responseData", {})
        .get("translatedText")
    )

    if not translated_text:
        raise RuntimeError(
            "Translation returned an empty response"
        )

    return translated_text


# --------------------------------------------------
# LOGIN MANAGER
# --------------------------------------------------

@login_manager.user_loader
def load_user(user_id):

    return User.query.get(int(user_id))


# --------------------------------------------------
# CHATBOT PROCESSING
# --------------------------------------------------

def clean_up_sentence(sentence):

    tokens = nltk.word_tokenize(sentence)

    return [
        lemmatizer.lemmatize(word.lower())
        for word in tokens
    ]


def bag_of_words(sentence):

    token_words = clean_up_sentence(sentence)

    bag = [
        1 if word in token_words else 0
        for word in words
    ]

    return np.array(bag)


def predict_class(sentence):

    normalized_sentence = sentence.lower().replace(
        "pythagorean",
        "pythagoras"
    )

    # Normalize common greeting translations.

    greeting_key = normalized_sentence.strip(
        " \t\r\n.!?,;:"
    )

    greeting_aliases = {
        "namasthe",
        "namaskara",
        "namaskara uncle",
        "welcome",
    }

    if greeting_key in greeting_aliases:
        normalized_sentence = "hello"

    bow = bag_of_words(normalized_sentence)

    res = model.predict(
        np.array([bow]),
        verbose=0
    )[0]

    ERROR_THRESHOLD = 0.25

    results = [
        [i, probability]
        for i, probability in enumerate(res)
        if probability > ERROR_THRESHOLD
    ]

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    predicted = [
        {
            "intent": classes[result[0]],
            "probability": str(result[1]),
        }
        for result in results
    ]

    # Targeted fallback for Pythagoras questions.

    lower_text = normalized_sentence.lower()

    if not predicted and "pythagor" in lower_text:

        if any(
            term in lower_text
            for term in [
                "theorem",
                "triangle",
                "hypotenuse",
                "formula",
            ]
        ):

            return [
                {
                    "intent": "pythagoras",
                    "probability": "rule-based",
                }
            ]

    return predicted


def get_response(intents_list, intents_json):

    if not intents_list:

        return (
            "I'm not sure about that. Try asking about Maths, "
            "Physics, Chemistry, Biology, or Computer Science!"
        )

    tag = intents_list[0]["intent"]

    for intent in intents_json["intents"]:

        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I'm not sure about that. Try asking about a subject!"


# --------------------------------------------------
# TEXT-TO-SPEECH
# --------------------------------------------------

def generate_audio(text, language="en"):

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    filename = f"output_{uuid.uuid4().hex}.mp3"

    mp3_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    tts = gTTS(
        text=text,
        lang=language,
        slow=False
    )

    tts.save(mp3_path)

    return mp3_path


# --------------------------------------------------
# LANDING PAGE
# --------------------------------------------------

@app.route("/")
def index():

    return render_template("landing.html")


# --------------------------------------------------
# REGISTRATION
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if current_user.is_authenticated:
        return redirect(url_for("chat"))

    if request.method == "POST":

        data = request.get_json() or {}

        if User.query.filter_by(
            email=data.get("email")
        ).first():

            return jsonify({
                "success": False,
                "message": "Email already registered.",
            })

        if User.query.filter_by(
            username=data.get("username")
        ).first():

            return jsonify({
                "success": False,
                "message": "Username already taken.",
            })

        user = User(
            full_name=data["full_name"],
            email=data["email"],
            username=data["username"],
            grade=data.get("grade", ""),
            school=data.get("school", ""),
        )

        user.set_password(data["password"])

        db.session.add(user)
        db.session.commit()

        login_user(user)

        return jsonify({
            "success": True,
            "redirect": url_for("chat"),
        })

    return render_template(
        "auth.html",
        mode="register"
    )


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if current_user.is_authenticated:
        return redirect(url_for("chat"))

    if request.method == "POST":

        data = request.get_json() or {}

        user = User.query.filter_by(
            username=data.get("username")
        ).first()

        if user and user.check_password(
            data.get("password", "")
        ):

            login_user(user)

            return jsonify({
                "success": True,
                "redirect": url_for("chat"),
            })

        return jsonify({
            "success": False,
            "message": "Invalid username or password.",
        })

    return render_template(
        "auth.html",
        mode="login"
    )


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(url_for("index"))


# --------------------------------------------------
# CHAT PAGE
# --------------------------------------------------

@app.route("/chat")
@login_required
def chat():

    return render_template(
        "chat.html",
        user=current_user
    )


# --------------------------------------------------
# CHAT MESSAGE API
# --------------------------------------------------

@app.route("/api/message", methods=["POST"])
@login_required
def api_message():

    data = request.get_json() or {}

    user_msg = data.get("message", "").strip()

    target_language = data.get(
        "language",
        "en"
    )

    sess_id = data.get(
        "session_id",
        "default"
    )

    supported_languages = {
        "en",
        "ta",
        "hi",
        "te",
        "ml",
        "kn",
        "fr",
        "es",
        "de",
        "ar",
    }

    if not user_msg:

        return jsonify({
            "error": "Empty message"
        }), 400

    if target_language not in supported_languages:

        return jsonify({
            "error": "Unsupported language"
        }), 400

    if not MODEL_LOADED:

        return jsonify({
            "error": "Chatbot model is not loaded. Please check the model files."
        }), 503

    try:

        # Detect the original input language.

        source_language = detect_input_language(
            user_msg,
            target_language
        )

        # Translate the user's message into English.

        english_message = translate_text(
            user_msg,
            target_language="en",
            source_language=source_language
        )

        app.logger.info(
            "Original message: %s",
            user_msg
        )

        app.logger.info(
            "Translated message: %s",
            english_message
        )

        # Process the English message.

        intents_list = predict_class(
            english_message
        )

        app.logger.info(
            "Predicted intents: %s",
            intents_list
        )

        english_reply = get_response(
            intents_list,
            intents
        )

        # Translate the response into the selected language.

        bot_reply = translate_text(
            english_reply,
            target_language=target_language,
            source_language="en"
        )

        # Generate audio.

        audio_url = None

        try:

            audio_path = generate_audio(
                bot_reply,
                language=target_language
            )

            audio_url = (
                f"/static/hologram/"
                f"{os.path.basename(audio_path)}"
            )

        except Exception:

            app.logger.exception(
                "Audio generation failed"
            )

        # Save the user's message.

        user_entry = ChatMessage(
            user_id=current_user.id,
            sender="user",
            message=user_msg,
            session_id=sess_id
        )

        # Save the chatbot response.

        bot_entry = ChatMessage(
            user_id=current_user.id,
            sender="bot",
            message=bot_reply,
            session_id=sess_id
        )

        db.session.add(user_entry)
        db.session.add(bot_entry)

        db.session.commit()

        return jsonify({
            "reply": bot_reply,
            "timestamp": datetime.utcnow().strftime(
                "%I:%M %p"
            ),
            "audio_url": audio_url,
            "video_url": None,
        })

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "Chat translation or response generation failed"
        )

        return jsonify({
            "error": "Unable to process your message. Please try again."
        }), 500


# --------------------------------------------------
# CONVERSATION HISTORY API
# --------------------------------------------------

@app.route("/api/history", methods=["GET"])
@login_required
def api_history():

    messages = (
        ChatMessage.query
        .filter_by(user_id=current_user.id)
        .order_by(
            ChatMessage.timestamp.asc(),
            ChatMessage.id.asc()
        )
        .all()
    )

    conversations = {}

    for message in messages:

        session_id = message.session_id or "default"

        if session_id not in conversations:

            conversations[session_id] = {
                "session_id": session_id,
                "title": "New Conversation",
                "timestamp": message.timestamp.strftime(
                    "%d %b %Y, %I:%M %p"
                ),
                "sort_time": message.timestamp,
            }

        if (
            message.sender == "user"
            and conversations[session_id]["title"] == "New Conversation"
        ):

            conversations[session_id]["title"] = (
                message.message[:60]
            )

    result = sorted(
        conversations.values(),
        key=lambda item: item["sort_time"],
        reverse=True
    )

    for conversation in result:
        conversation.pop("sort_time")

    return jsonify({
        "conversations": result
    })


@app.route("/api/history/<session_id>", methods=["GET"])
@login_required
def get_conversation(session_id):

    from sqlalchemy import or_

    query = ChatMessage.query.filter_by(
        user_id=current_user.id
    )

    if session_id == "default":

        query = query.filter(
            or_(
                ChatMessage.session_id == "default",
                ChatMessage.session_id.is_(None)
            )
        )

    else:

        query = query.filter_by(
            session_id=session_id
        )

    messages = query.order_by(
        ChatMessage.timestamp.asc(),
        ChatMessage.id.asc()
    ).all()

    return jsonify({
        "messages": [
            {
                "sender": message.sender,
                "message": message.message,
                "timestamp": message.timestamp.strftime(
                    "%I:%M %p"
                ),
            }
            for message in messages
        ]
    })


# --------------------------------------------------
# CLEAR CHAT HISTORY
# --------------------------------------------------

@app.route("/api/clear_history", methods=["POST"])
@login_required
def clear_history():

    ChatMessage.query.filter_by(
        user_id=current_user.id
    ).delete()

    db.session.commit()

    return jsonify({
        "success": True
    })


# --------------------------------------------------
# PDF EXPORT
# --------------------------------------------------

@app.route("/api/export_pdf")
@login_required
def export_pdf():

    messages = (
        ChatMessage.query
        .filter_by(user_id=current_user.id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=20,
    )

    user_style = ParagraphStyle(
        "User",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#1d4ed8"),
        backColor=colors.HexColor("#eff6ff"),
        borderPadding=(8, 10, 8, 10),
        spaceAfter=6,
        leftIndent=60,
    )

    bot_style = ParagraphStyle(
        "Bot",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#1a1a2e"),
        backColor=colors.HexColor("#f8fafc"),
        borderPadding=(8, 10, 8, 10),
        spaceAfter=6,
        rightIndent=60,
    )

    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#9ca3af"),
        spaceAfter=2,
    )

    story = []

    story.append(
        Paragraph(
            "EduBot Chat History",
            title_style
        )
    )

    story.append(
        Paragraph(
            (
                f"Student: {current_user.full_name}  |  "
                f"Exported: {datetime.utcnow().strftime('%B %d, %Y %I:%M %p')}"
            ),
            subtitle_style,
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#e5e7eb"),
            spaceAfter=16
        )
    )

    for message in messages:

        timestamp = message.timestamp.strftime(
            "%b %d, %I:%M %p"
        )

        if message.sender == "user":

            story.append(
                Paragraph(
                    f"You {timestamp}",
                    label_style
                )
            )

            story.append(
                Paragraph(
                    message.message,
                    user_style
                )
            )

        else:

            story.append(
                Paragraph(
                    f"EduBot {timestamp}",
                    label_style
                )
            )

            story.append(
                Paragraph(
                    message.message,
                    bot_style
                )
            )

        story.append(
            Spacer(1, 4)
        )

    doc.build(story)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=(
            f"edubot_chat_{current_user.username}.pdf"
        ),
        mimetype="application/pdf",
    )


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=3000,
        debug=True
    )
