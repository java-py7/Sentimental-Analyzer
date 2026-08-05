# ------------------- Importing Libraries -------------------
from flask import Flask, render_template, request, jsonify, redirect, url_for
from collections import Counter
import googleapiclient.discovery
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from datetime import datetime
import uuid
from textblob import TextBlob
import time
from dotenv import load_dotenv
from database import db
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from authlib.integrations.flask_client import OAuth
from models import User, Analysis, Comment
from flask_login import login_user, logout_user, current_user, login_required
import base64
from authlib.integrations.base_client.errors import OAuthError
from datetime import timedelta

app = Flask(__name__)

load_dotenv()

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

app.config["REMEMBER_COOKIE_DURATION"] = timedelta(days=30)
app.config["REMEMBER_COOKIE_SECURE"] = True
app.config["REMEMBER_COOKIE_HTTPONLY"] = True
app.config["REMEMBER_COOKIE_SAMESITE"] = "Lax"

# ------------------- Database Configuration -------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

bcrypt = Bcrypt(app)

login_manager = LoginManager(app)

login_manager.login_view = "google_login"

login_manager.login_message = "Please login first."

login_manager.login_message_category = "warning"

oauth = OAuth(app)

oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile"
    }
)


# Latest analysis (used for exporting)
LATEST_ANALYSIS = {}


# ------------------- Sentiment Analysis -------------------

def analyze_sentiments_enhanced(comments):
    """Enhanced sentiment analysis using VADER + TextBlob + keyword rules"""

    sentiments = []
    analyzer = SentimentIntensityAnalyzer()

    strong_positive = {
        "amazing", "awesome", "excellent", "fantastic",
        "brilliant", "outstanding", "incredible",
        "wonderful", "perfect", "phenomenal",
        "superb", "magnificent"
    }

    positive_words = {
        "love", "great", "good", "nice", "beautiful",
        "best", "thank", "thanks", "happy", "enjoy",
        "favorite", "impressive", "cool", "sweet",
        "fire", "lit", "blessed", "grateful"
    }

    strong_negative = {
        "hate", "terrible", "awful", "horrible",
        "disgusting", "pathetic", "trash",
        "garbage", "useless", "worthless",
        "stupid", "idiotic", "moronic", "dumb"
    }

    negative_words = {
        "bad", "worst", "sucks", "annoying",
        "irritating", "boring", "lame",
        "disappointed", "upset", "angry",
        "mad", "pissed", "fuck", "shit",
        "damn", "hell", "asshole", "bitch",
        "crap", "fail"
    }

    negations = {
        "not", "no", "never", "none",
        "don't", "doesn't", "didn't",
        "won't", "can't"
    }

    for comment in comments:

        original_comment = comment.strip()

        if not original_comment:
            continue

        text = original_comment.lower()

        # ---------- VADER ----------
        vader = analyzer.polarity_scores(original_comment)["compound"]

        # ---------- TextBlob ----------
        try:
            blob = TextBlob(original_comment).sentiment.polarity
        except:
            blob = 0

        score = (vader * 0.75) + (blob * 0.25)

        words = text.split()

        pos = 0
        neg = 0

        for i, word in enumerate(words):

            previous = words[max(0, i - 2):i]

            negated = any(w in negations for w in previous)

            if word in strong_positive:
                pos += 2 if not negated else -2

            elif word in positive_words:
                pos += 1 if not negated else -1

            elif word in strong_negative:
                neg += 2 if not negated else -2

            elif word in negative_words:
                neg += 1 if not negated else -1

        # ---------- Final Decision ----------

        if pos > neg:

            if score >= 0.05 or pos >= 2:
                sentiment = "Positive"
            else:
                sentiment = "Neutral"

        elif neg > pos:

            if score <= -0.05 or neg >= 2:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

        else:

            if score >= 0.15:
                sentiment = "Positive"

            elif score <= -0.15:
                sentiment = "Negative"

            else:
                sentiment = "Neutral"

        # Obvious phrases

        if any(p in text for p in [
            "i love",
            "so good",
            "really good",
            "very good"
        ]):
            sentiment = "Positive"

        elif any(p in text for p in [
            "i hate",
            "so bad",
            "really bad",
            "very bad",
            "fuck this"
        ]):
            sentiment = "Negative"

        sentiments.append((sentiment, original_comment))

    return sentiments


# ------------------- Fetching YouTube Comments -------------------
def fetch_comments(api_key, video_id, max_results=200):
    """Fetch YouTube comments with specified limit"""
    comments = []
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    # FETCH VIDEO TITLE
    video_response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    video_title = video_response["items"][0]["snippet"]["title"]

    # Determine the batch size for API requests (YouTube API max is 100)
    batch_size = min(100, max_results)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=batch_size,
        textFormat="plainText"
    )
    response = request.execute()

    print(f"Fetching up to {max_results} comments...")

    while response and len(comments) < max_results:
        batch_comments = []
        for item in response.get("items", []):
            if len(comments) >= max_results:
                break
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            # Keep every non-empty comment
            if comment.strip():
                comments.append(comment)
                batch_comments.append(comment)

        print(f"Fetched {len(batch_comments)} comments (Total: {len(comments)})")

        # Check if we need more comments and if there's a next page
        if "nextPageToken" in response and len(comments) < max_results:
            remaining = max_results - len(comments)
            next_batch_size = min(100, remaining)

            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=next_batch_size,
                pageToken=response["nextPageToken"],
                textFormat="plainText"
            )
            response = request.execute()
        else:
            break

    print(f"✅ Successfully fetched {len(comments)} comments")
    return comments[:max_results], video_title  # Ensure we don't exceed the limit


# ------------------- Sentiment Analysis -------------------
def analyze_sentiments(comments):
    sentiments = []
    analyzer = SentimentIntensityAnalyzer()
    for comment in comments:
        vs = analyzer.polarity_scores(comment)
        compound_score = vs['compound']
        if compound_score >= 0.05:
            sentiments.append(("Positive", comment))
        elif compound_score <= -0.05:
            sentiments.append(("Negative", comment))
        else:
            sentiments.append(("Neutral", comment))
    return sentiments


# ------------------- Flask Routes -------------------

@app.route("/login/google")
def google_login():

    redirect_uri = url_for(
        "google_authorized",
        _external=True
    )

    return oauth.google.authorize_redirect(
        redirect_uri
    )

@app.route("/login/google/authorized")
def google_authorized():

    try:
        token = oauth.google.authorize_access_token()

    except OAuthError:
        return redirect(url_for("home", toast="login_cancelled"))

    user_info = token["userinfo"]
    google_id = user_info["sub"]
    name = user_info["name"]
    email = user_info["email"]
    picture = user_info.get("picture")
    user = User.query.filter_by(
        google_id=google_id
    ).first()

    if not user:
        user = User(
            google_id=google_id,
            name=name,
            email=email,
            profile_picture=picture
        )

        db.session.add(user)
        db.session.commit()

    login_user(user, remember=True)

    return redirect(url_for("home", login="success"))

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(url_for("home", logout="success"))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    global LATEST_ANALYSIS
    api_key = os.getenv("YOUTUBE_API_KEY")
    video_url = request.form['video_url']
    max_comments_param = request.form.get('max_comments', '200')

    # Handle max_comments parameter
    if max_comments_param == 'all':
        max_comments = 2000  # Set a reasonable upper limit
    else:
        try:
            max_comments = int(max_comments_param)
            # Ensure reasonable limits
            max_comments = min(max(max_comments, 10), 2000)  # Between 10 and 2000
        except (ValueError, TypeError):
            max_comments = 200  # Default fallback

    # Extract video ID from URL
    video_id = None

    # Handle standard YouTube URLs (youtube.com/watch?v=)
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    # Handle shortened YouTube URLs (youtu.be/)
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[1].split("?")[0]

    if not video_id:
        return jsonify({
            "success":False,
            "message":"Invalid YouTube URL."
        }),400

    try:
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add processing delay for preloader (easily adjustable)
        time.sleep(2)  # Initial delay

        comments, video_title = fetch_comments(
            api_key,
            video_id,
            max_comments
        )

        # Processing delay
        time.sleep(1)
        sentiments = analyze_sentiments_enhanced(comments)
        sentiment_counts = Counter([s for s, c in sentiments])

        LATEST_ANALYSIS = {

            "video_title": video_title,

            "video_id": video_id,

            "video_url": video_url,

            "total_comments": len(comments),

            "positive": sentiment_counts.get("Positive",0),

            "negative": sentiment_counts.get("Negative",0),

            "neutral": sentiment_counts.get("Neutral",0),

            "comments": sentiments

        }

        analysis_db_id = None

    # ---------------- Save Analysis ----------------

        if current_user.is_authenticated:

            analysis = Analysis(

                user_id=current_user.id,

                video_title=video_title,

                video_url=video_url,

                video_id=video_id,

                positive=sentiment_counts.get("Positive",0),

                negative=sentiment_counts.get("Negative",0),

                neutral=sentiment_counts.get("Neutral",0),

                total_comments=len(comments)

            )

            db.session.add(analysis)

            db.session.commit()


    # ---------------- Save Comments ----------------

            for sentiment, comment in sentiments:

                db.session.add(

                    Comment(

                        analysis_id=analysis.id,

                        sentiment=sentiment,

                        comment=comment

                    )

                )

            db.session.commit()

            analysis_db_id = analysis.id



        # Analysis delay
        time.sleep(1.5)

        # Visualization delay
        time.sleep(1)

        # Final processing delay
        time.sleep(0.5)

        print(f"▶ Fetching comments for: {video_id}")
        print(f"Total comments fetched: {len(comments)}")
        print(f"Sentiment counts: {sentiment_counts}")
        print(f"Analysis ID: {analysis_id}")

        return jsonify({

            "success": True,

            "video_title": video_title,

            "video_url": video_url,

            "analysis_id": analysis_id,

            "database_id": analysis_db_id,

            "max_comments_requested": max_comments_param,

            "summary": {
                "positive": sentiment_counts.get("Positive", 0),
                "negative": sentiment_counts.get("Negative", 0),
                "neutral": sentiment_counts.get("Neutral", 0)
            },

            "comments": [
                {
                    "sentiment": sentiment,
                    "comment": comment
                }
                for sentiment, comment in sentiments
            ]

        })

    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }),500


@app.route("/save-images", methods=["POST"])
@login_required
def save_images():

    data = request.get_json()

    analysis = Analysis.query.get(
        data["analysis_id"]
    )

    if not analysis:

        return jsonify({
            "success": False
        }), 404

    chart = data["chart_image"]

    wordcloud = data["wordcloud_image"]

    if chart.startswith("data:image"):

        chart = chart.split(",")[1]

    if wordcloud.startswith("data:image"):

        wordcloud = wordcloud.split(",")[1]

    analysis.chart_image = base64.b64decode(chart)

    analysis.wordcloud_image = base64.b64decode(wordcloud)

    db.session.commit()

    return jsonify({
        "success": True
    })



@app.route("/get-history")
def get_history():

    if not current_user.is_authenticated:
        return jsonify([])

    analyses = Analysis.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Analysis.created_at.desc()
    ).all()

    history = []

    for analysis in analyses:

        history.append({

            "id": analysis.id,

            "title": analysis.video_title,

            "video_id": analysis.video_id,

            "created_at": analysis.created_at.strftime("%d %b %Y"),

            "positive": analysis.positive,

            "negative": analysis.negative,

            "neutral": analysis.neutral,

            "total_comments": analysis.total_comments

        })

    return jsonify(history)



@app.route("/get-analysis/<int:analysis_id>")
@login_required
def get_analysis(analysis_id):

    analysis = Analysis.query.filter_by(
        id=analysis_id,
        user_id=current_user.id
    ).first()

    if not analysis:

        return jsonify({
            "success":False
        }),404

    comments = []

    for item in analysis.comments:

        comments.append({

            "sentiment":item.sentiment,

            "comment":item.comment

        })

    return jsonify({

    "success": True,

    "analysis": {

        "id": analysis.id,

        "video_title": analysis.video_title,

        "video_url": analysis.video_url,

        "video_id": analysis.video_id,

        "created_at": analysis.created_at.strftime("%d %b %Y"),

        "total_comments": analysis.total_comments,

        "summary": {

            "positive": analysis.positive,

            "negative": analysis.negative,

            "neutral": analysis.neutral

        },

        "comments": comments,

        "chart": "data:image/png;base64," +
            base64.b64encode(analysis.chart_image).decode(),

        "wordcloud": "data:image/png;base64," +
            base64.b64encode(analysis.wordcloud_image).decode()

    }

})


@app.route("/delete-analysis/<int:analysis_id>",methods=["DELETE"])
@login_required
def delete_analysis(analysis_id):

    analysis = Analysis.query.filter_by(

        id=analysis_id,

        user_id=current_user.id

    ).first()

    if not analysis:

        return jsonify({

            "success":False

        }),404

    db.session.delete(analysis)

    db.session.commit()

    return jsonify({

        "success":True

    })


@app.route("/clear-history", methods=["DELETE"])
@login_required
def clear_history():

    analyses = Analysis.query.filter_by(
        user_id=current_user.id
    ).all()

    for analysis in analyses:

        db.session.delete(analysis)

    db.session.commit()

    return jsonify({
        "success": True
    })


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)
