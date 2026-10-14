# ------------------- Importing Libraries -------------------
import sys
import os

# Ensure console output handles Unicode cleanly on Windows without charmap encoding crashes
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

# Ensure project root is in sys.path so modules resolve in any IDE or CLI environment
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import base64
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    current_user,
    login_required
)
from flask_bcrypt import Bcrypt
from authlib.integrations.flask_client import OAuth
from authlib.integrations.base_client.errors import OAuthError

import googleapiclient.discovery
from googleapiclient.errors import HttpError

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from dotenv import load_dotenv

from database import db
from models import User, Analysis, Comment

try:
    nltk.download("vader_lexicon", quiet=True)
except Exception:
    pass

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
        except Exception:
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


def extract_video_id(video_url):
    """Extract YouTube video ID from various URL formats"""
    if not video_url:
        return None
    url = str(video_url).strip()
    if "v=" in url:
        return url.split("v=")[1].split("&")[0].split("?")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    elif "youtube.com/shorts/" in url:
        return url.split("youtube.com/shorts/")[1].split("?")[0].split("&")[0]
    elif "youtube.com/embed/" in url:
        return url.split("youtube.com/embed/")[1].split("?")[0].split("&")[0]
    elif len(url) == 11 and url.isalnum():
        return url
    return None


def parse_comment_limit(param, default=200):
    """Parse and clamp comment limit between 10 and 2000"""
    if param == 'all':
        return 2000
    try:
        val = int(param)
        return min(max(val, 10), 2000)
    except (ValueError, TypeError):
        return default


# ------------------- Fetching YouTube Comments -------------------
def fetch_comments(api_key, video_id, max_results=200):
    """Fetch YouTube comments with specified limit and resilient error handling"""
    comments = []
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    # FETCH VIDEO TITLE AND DETAILS
    try:
        video_response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()
    except HttpError as err:
        err_str = str(err)
        if "quotaExceeded" in err_str:
            raise Exception("YouTube API quota exceeded. Please try again tomorrow.")
        elif "videoNotFound" in err_str:
            raise Exception(f"Video not found (Video ID: {video_id})")
        raise Exception(f"YouTube API error: {err}")

    items = video_response.get("items", [])
    if not items:
        raise Exception(f"Video not found or is private (Video ID: {video_id})")

    video_title = items[0]["snippet"]["title"]

    # Determine the batch size for API requests (YouTube API max is 100)
    batch_size = min(100, max_results)

    try:
        request_obj = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=batch_size,
            textFormat="plainText"
        )
        response = request_obj.execute()
    except HttpError as err:
        err_str = str(err)
        if "commentsDisabled" in err_str:
            raise Exception(f"Comments are disabled on this video (\"{video_title}\").")
        elif "quotaExceeded" in err_str:
            raise Exception("YouTube API quota exceeded. Please try again tomorrow.")
        raise Exception(f"Could not load comments: {err}")

    print(f"Fetching up to {max_results} comments for video ID: {video_id}...")

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

            try:
                request_obj = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=next_batch_size,
                    pageToken=response["nextPageToken"],
                    textFormat="plainText"
                )
                response = request_obj.execute()
            except Exception:
                break
        else:
            break

    print(f"Successfully fetched {len(comments)} comments")
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
    video_url = request.form.get('video_url', '').strip()
    max_comments_param = request.form.get('max_comments', '200')
    max_comments = parse_comment_limit(max_comments_param)

    # Extract video ID from URL using unified extractor
    video_id = extract_video_id(video_url)

    if not video_id:
        return jsonify({
            "success": False,
            "message": "Invalid YouTube URL. Please provide a valid YouTube watch, shorts, or youtu.be link."
        }), 400

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

        print(f"Fetching comments for: {video_id}")
        print(f"Total comments fetched: {len(comments)}")
        print(f"Sentiment counts: {sentiment_counts}")
        print(f"Analysis ID: {analysis_id}")

        return jsonify({

            "success": True,

            "video_title": video_title,

            "video_url": video_url,

            "video_id": video_id,

            "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",

            "analysis_id": analysis_id,
            "timestamp": timestamp,
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


# ------------------- Compare Two YouTube Videos -------------------
@app.route('/compare', methods=['POST'])
def compare():
    api_key = os.getenv("YOUTUBE_API_KEY")
    url_a = request.form.get('video_url_a', '').strip()
    url_b = request.form.get('video_url_b', '').strip()
    max_comments_a = request.form.get('max_comments_a', '200')
    max_comments_b = request.form.get('max_comments_b', '200')

    limit_a = parse_comment_limit(max_comments_a)
    limit_b = parse_comment_limit(max_comments_b)

    vid_a = extract_video_id(url_a)
    vid_b = extract_video_id(url_b)

    if not vid_a or not vid_b:
        return jsonify({
            "success": False,
            "message": "Please provide valid YouTube URLs for both Video A and Video B."
        }), 400

    if vid_a == vid_b:
        return jsonify({
            "success": False,
            "message": "Both URLs identify the same video. Please enter two distinct videos to compare."
        }), 400

    try:
        # Fetch real comments and video titles using existing backend function
        comments_a, title_a = fetch_comments(api_key, vid_a, limit_a)
        comments_b, title_b = fetch_comments(api_key, vid_b, limit_b)

        if not comments_a:
            return jsonify({
                "success": False,
                "message": f"No comments found for Video A: '{title_a}'. Comments may be disabled or empty."
            }), 400

        if not comments_b:
            return jsonify({
                "success": False,
                "message": f"No comments found for Video B: '{title_b}'. Comments may be disabled or empty."
            }), 400

        # Run existing enhanced sentiment analysis
        sentiments_a = analyze_sentiments_enhanced(comments_a)
        sentiments_b = analyze_sentiments_enhanced(comments_b)

        counts_a = Counter([s for s, c in sentiments_a])
        counts_b = Counter([s for s, c in sentiments_b])

        total_a = len(comments_a)
        total_b = len(comments_b)

        pos_a = counts_a.get("Positive", 0)
        neg_a = counts_a.get("Negative", 0)
        neu_a = counts_a.get("Neutral", 0)

        pos_b = counts_b.get("Positive", 0)
        neg_b = counts_b.get("Negative", 0)
        neu_b = counts_b.get("Neutral", 0)

        pct_pos_a = round((pos_a / total_a) * 100, 1) if total_a > 0 else 0
        pct_neg_a = round((neg_a / total_a) * 100, 1) if total_a > 0 else 0
        pct_neu_a = round((neu_a / total_a) * 100, 1) if total_a > 0 else 0

        pct_pos_b = round((pos_b / total_b) * 100, 1) if total_b > 0 else 0
        pct_neg_b = round((neg_b / total_b) * 100, 1) if total_b > 0 else 0
        pct_neu_b = round((neu_b / total_b) * 100, 1) if total_b > 0 else 0

        # Exact calculated percentage point differences (A minus B)
        diff_pos = round(pct_pos_a - pct_pos_b, 1)
        diff_neg = round(pct_neg_a - pct_neg_b, 1)
        diff_neu = round(pct_neu_a - pct_neu_b, 1)

        # Factual comparison leader identification
        if pct_pos_a > pct_pos_b:
            lead_video = "Video A"
            lead_title = title_a
            diff_text = f"Video A has a {abs(diff_pos)}% higher positive sentiment share than Video B."
        elif pct_pos_b > pct_pos_a:
            lead_video = "Video B"
            lead_title = title_b
            diff_text = f"Video B has a {abs(diff_pos)}% higher positive sentiment share than Video A."
        else:
            lead_video = "Tied"
            lead_title = "Both videos"
            diff_text = "Both videos have an identical positive sentiment percentage."

        summary_text = (
            f"Based on real sentiment analysis of {total_a} comments for \"{title_a}\" and {total_b} comments for \"{title_b}\": "
            f"{diff_text} Video A shows {pct_pos_a}% positive, {pct_neg_a}% negative, and {pct_neu_a}% neutral reactions. "
            f"Video B shows {pct_pos_b}% positive, {pct_neg_b}% negative, and {pct_neu_b}% neutral reactions."
        )

        # Save both analyses to SQLite database if authenticated so they appear in history
        if current_user.is_authenticated:
            for v_title, v_url, v_id, p, n, nu, t, sents in [
                (title_a, url_a, vid_a, pos_a, neg_a, neu_a, total_a, sentiments_a),
                (title_b, url_b, vid_b, pos_b, neg_b, neu_b, total_b, sentiments_b)
            ]:
                analysis_record = Analysis(
                    user_id=current_user.id,
                    video_title=v_title,
                    video_url=v_url,
                    video_id=v_id,
                    positive=p,
                    negative=n,
                    neutral=nu,
                    total_comments=t
                )
                db.session.add(analysis_record)
                db.session.commit()

                for sentiment, comment in sents:
                    db.session.add(Comment(
                        analysis_id=analysis_record.id,
                        sentiment=sentiment,
                        comment=comment
                    ))
                db.session.commit()

        return jsonify({
            "success": True,
            "video_a": {
                "title": title_a,
                "url": url_a,
                "video_id": vid_a,
                "thumbnail": f"https://img.youtube.com/vi/{vid_a}/hqdefault.jpg",
                "total_comments": total_a,
                "positive": pos_a,
                "negative": neg_a,
                "neutral": neu_a,
                "pct_positive": pct_pos_a,
                "pct_negative": pct_neg_a,
                "pct_neutral": pct_neu_a,
                "sample_comments": [{"sentiment": s, "comment": c} for s, c in sentiments_a[:8]]
            },
            "video_b": {
                "title": title_b,
                "url": url_b,
                "video_id": vid_b,
                "thumbnail": f"https://img.youtube.com/vi/{vid_b}/hqdefault.jpg",
                "total_comments": total_b,
                "positive": pos_b,
                "negative": neg_b,
                "neutral": neu_b,
                "pct_positive": pct_pos_b,
                "pct_negative": pct_neg_b,
                "pct_neutral": pct_neu_b,
                "sample_comments": [{"sentiment": s, "comment": c} for s, c in sentiments_b[:8]]
            },
            "comparison": {
                "diff_positive": diff_pos,
                "diff_negative": diff_neg,
                "diff_neutral": diff_neu,
                "lead_video": lead_video,
                "lead_title": lead_title,
                "summary": summary_text
            }
        })
    except Exception as e:
        print(f"Compare error: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ------------------- Dashboard Real Statistics -------------------
@app.route('/dashboard-data')
def dashboard_data():
    if not current_user.is_authenticated:
        return jsonify({
            "authenticated": False,
            "total_analyses": 0,
            "total_comments": 0,
            "avg_positive": 0,
            "avg_negative": 0,
            "avg_neutral": 0,
            "recent_analyses": []
        })

    analyses = Analysis.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Analysis.created_at.desc()
    ).all()

    total_analyses = len(analyses)
    if total_analyses == 0:
        return jsonify({
            "authenticated": True,
            "total_analyses": 0,
            "total_comments": 0,
            "avg_positive": 0,
            "avg_negative": 0,
            "avg_neutral": 0,
            "recent_analyses": []
        })

    total_comments = sum((a.total_comments or 0) for a in analyses)
    total_pos = sum((a.positive or 0) for a in analyses)
    total_neg = sum((a.negative or 0) for a in analyses)
    total_neu = sum((a.neutral or 0) for a in analyses)

    avg_pos = round((total_pos / total_comments * 100), 1) if total_comments > 0 else 0
    avg_neg = round((total_neg / total_comments * 100), 1) if total_comments > 0 else 0
    avg_neu = round((total_neu / total_comments * 100), 1) if total_comments > 0 else 0

    recent = []
    for a in analyses[:6]:
        tot = (a.positive or 0) + (a.negative or 0) + (a.neutral or 0)
        p_pct = round(((a.positive or 0) / tot * 100), 1) if tot > 0 else 0
        recent.append({
            "id": a.id,
            "title": a.video_title,
            "video_id": a.video_id,
            "thumbnail": f"https://img.youtube.com/vi/{a.video_id}/hqdefault.jpg" if a.video_id else "",
            "date": a.created_at.strftime("%d %b %Y"),
            "total_comments": a.total_comments,
            "positive": a.positive,
            "negative": a.negative,
            "neutral": a.neutral,
            "pct_positive": p_pct
        })

    return jsonify({
        "authenticated": True,
        "total_analyses": total_analyses,
        "total_comments": total_comments,
        "total_positive": total_pos,
        "total_negative": total_neg,
        "total_neutral": total_neu,
        "avg_positive": avg_pos,
        "avg_negative": avg_neg,
        "avg_neutral": avg_neu,
        "recent_analyses": recent
    })


@app.route("/save-images", methods=["POST"])
def save_images():
    if not current_user.is_authenticated:
        return jsonify({"success": False, "message": "Authentication required"}), 401

    data = request.get_json(silent=True) or {}
    analysis_id = data.get("analysis_id")
    if not analysis_id:
        return jsonify({"success": False, "message": "Missing analysis ID"}), 400

    analysis = db.session.get(Analysis, analysis_id)
    if not analysis or analysis.user_id != current_user.id:
        return jsonify({"success": False, "message": "Analysis not found"}), 404

    chart = data.get("chart_image")
    wordcloud = data.get("wordcloud_image")

    try:
        # Partial update: only update chart if a valid data:image string is provided
        if chart and isinstance(chart, str) and chart.startswith("data:image"):
            chart_base64 = chart.split(",")[1]
            analysis.chart_image = base64.b64decode(chart_base64)

        # Partial update: only update wordcloud if a valid data:image string is provided
        if wordcloud and isinstance(wordcloud, str) and wordcloud.startswith("data:image"):
            wc_base64 = wordcloud.split(",")[1]
            analysis.wordcloud_image = base64.b64decode(wc_base64)

        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500



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
def get_analysis(analysis_id):
    if not current_user.is_authenticated:
        return jsonify({"success": False, "message": "Authentication required"}), 401

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

        "chart": ("data:image/png;base64," + base64.b64encode(analysis.chart_image).decode()) if analysis.chart_image else "",

        "wordcloud": ("data:image/png;base64," + base64.b64encode(analysis.wordcloud_image).decode()) if analysis.wordcloud_image else ""

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
    try:
        return db.session.get(User, int(user_id))
    except (ValueError, TypeError):
        return None

with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)
