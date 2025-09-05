# ------------------- Importing Libraries -------------------
from flask import Flask, render_template, request
from collections import Counter
import googleapiclient.discovery
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import json
import os
from datetime import datetime
import uuid
import re
from textblob import TextBlob
import time
import gc

app = Flask(__name__)

# History file path
HISTORY_FILE = 'analysis_history.json'


# ------------------- History Management -------------------
def load_history():
    """Load analysis history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_to_history(video_url, video_id, sentiment_counts, total_comments, analysis_id, timestamp, max_comments=None):
    """Save analysis result to history"""
    history = load_history()

    # Create new history entry
    entry = {
        'id': analysis_id,
        'timestamp': timestamp,
        'video_url': video_url,
        'video_id': video_id,
        'total_comments': total_comments,
        'max_comments_requested': max_comments,
        'sentiment_counts': dict(sentiment_counts),
        'wordcloud_file': f'wordcloud_{analysis_id}.png',
        'barchart_file': f'bar_chart_{analysis_id}.png'
    }

    # Add to beginning of list
    history.insert(0, entry)

    # Keep only last 50 analyses to prevent file from getting too large
    history = history[:50]

    # Save to file
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving history: {e}")


def delete_from_history(analysis_id):
    """Delete a specific analysis from history"""
    try:
        history = load_history()

        # Find and remove the entry with matching ID
        updated_history = [entry for entry in history if entry['id'] != analysis_id]

        if len(updated_history) < len(history):
            # Save updated history
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(updated_history, f, indent=2, ensure_ascii=False)

            # Try to delete associated image files
            try:
                os.remove(f'static/wordcloud_{analysis_id}.png')
            except:
                pass
            try:
                os.remove(f'static/bar_chart_{analysis_id}.png')
            except:
                pass

            print(f"Deleted analysis {analysis_id} from history")
            return True
        else:
            print(f"Analysis {analysis_id} not found in history")
            return False

    except Exception as e:
        print(f"Error deleting from history: {e}")
        return False


def clear_all_history():
    """Clear all history and associated files"""
    try:
        # Delete all chart files
        import glob
        for file_path in glob.glob('static/wordcloud_*.png'):
            try:
                os.remove(file_path)
            except:
                pass
        for file_path in glob.glob('static/bar_chart_*.png'):
            try:
                os.remove(file_path)
            except:
                pass

        # Clear history file
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)

        print("All history cleared")
        return True

    except Exception as e:
        print(f"Error clearing history: {e}")
        return False


# ------------------- Comment Cleaning -------------------
def remove_emojis(text):
    """Remove emojis and other Unicode symbols from text"""
    # Emoji patterns
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def clean_comment(comment):
    """Enhanced comment cleaning while preserving bad words for sentiment analysis"""
    if not comment or len(comment.strip()) == 0:
        return ""

    # Remove emojis first
    comment = remove_emojis(comment)

    # Remove URLs
    comment = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', comment)

    # Remove mentions and hashtags (but keep the text after #)
    comment = re.sub(r'@\w+', '', comment)
    comment = re.sub(r'#(\w+)', r'\1', comment)

    # Remove excessive punctuation (more than 2 consecutive)
    comment = re.sub(r'([!?.,:;])\1{2,}', r'\1\1', comment)

    # Remove standalone numbers but keep numbers within words
    comment = re.sub(r'\b\d+\b', '', comment)

    # Remove excessive whitespace
    comment = re.sub(r'\s+', ' ', comment)

    # Remove very short words but keep important ones
    words = comment.split()
    cleaned_words = []
    keep_short = {'i', 'a', 'is', 'it', 'be', 'to', 'of', 'in', 'on', 'at', 'up', 'so', 'no', 'go', 'me', 'we', 'my',
                  'ok', 'am', 'or', 'if', 'do'}

    for word in words:
        word_clean = word.lower().strip('.,!?;:')
        if len(word_clean) >= 2 or word_clean in keep_short:
            if word_clean and not word_clean.isdigit():  # Skip pure numbers
                cleaned_words.append(word)

    comment = ' '.join(cleaned_words)

    # Remove leading/trailing whitespace and punctuation
    comment = comment.strip(' .,!?;:')

    # Return only if meaningful content remains
    return comment if len(comment.strip()) > 2 else ""


# ------------------- Sentiment Analysis -------------------
def analyze_sentiments_enhanced(comments):
    """Enhanced sentiment analysis with multiple methods and better accuracy"""
    sentiments = []
    analyzer = SentimentIntensityAnalyzer()

    for comment in comments:
        # Clean the comment but preserve bad words for accurate sentiment
        cleaned_comment = clean_comment(comment)

        if not cleaned_comment:  # Skip empty comments
            continue

        # Normalize for better analysis
        normalized_comment = cleaned_comment.lower()

        # VADER analysis
        vs = analyzer.polarity_scores(cleaned_comment)
        vader_score = vs['compound']

        # TextBlob analysis for additional accuracy
        try:
            blob = TextBlob(cleaned_comment)
            textblob_score = blob.sentiment.polarity
        except:
            textblob_score = 0

        # Combine both methods with weighted average (VADER is better for social media)
        combined_score = (vader_score * 0.75) + (textblob_score * 0.25)

        # Enhanced word lists for better detection
        strong_positive = ['amazing', 'awesome', 'excellent', 'fantastic', 'brilliant', 'outstanding',
                           'incredible', 'wonderful', 'perfect', 'phenomenal', 'superb', 'magnificent']

        positive_indicators = ['love', 'great', 'good', 'nice', 'beautiful', 'best', 'thank', 'thanks',
                               'appreciate', 'glad', 'happy', 'enjoy', 'enjoyed', 'favorite', 'impressive',
                               'cool', 'sweet', 'dope', 'fire', 'lit', 'blessed', 'grateful']

        strong_negative = ['hate', 'terrible', 'awful', 'horrible', 'disgusting', 'pathetic', 'trash',
                           'garbage', 'useless', 'worthless', 'stupid', 'idiotic', 'moronic', 'dumb']

        negative_indicators = ['bad', 'worst', 'sucks', 'annoying', 'irritating', 'boring', 'lame',
                               'disappointed', 'disappoint', 'upset', 'angry', 'mad', 'pissed', 'fuck',
                               'shit', 'damn', 'hell', 'asshole', 'bitch', 'crap', 'suck', 'fail']

        # Negation handling
        negation_words = ['not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nobody',
                          'hardly', 'barely', 'scarcely', "don't", "doesn't", "didn't", "won't", "can't"]

        # Count indicators
        strong_pos_count = sum(1 for word in strong_positive if word in normalized_comment)
        pos_count = sum(1 for word in positive_indicators if word in normalized_comment)
        strong_neg_count = sum(1 for word in strong_negative if word in normalized_comment)
        neg_count = sum(1 for word in negative_indicators if word in normalized_comment)

        # Check for negations near positive words
        words = normalized_comment.split()
        negated_positive = 0
        negated_negative = 0

        for i, word in enumerate(words):
            # Check if there's a negation within 2 words before
            negation_nearby = any(neg in words[max(0, i - 2):i] for neg in negation_words)

            if word in positive_indicators and negation_nearby:
                negated_positive += 1
            elif word in negative_indicators and negation_nearby:
                negated_negative += 1

        # Adjust counts for negations
        effective_pos = (strong_pos_count * 2) + pos_count - negated_positive
        effective_neg = (strong_neg_count * 2) + neg_count - negated_negative

        # Determine sentiment with enhanced logic
        if effective_neg > effective_pos:
            if effective_neg >= 2 or combined_score <= -0.2:
                sentiment = "Negative"
            elif combined_score <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
        elif effective_pos > effective_neg:
            if effective_pos >= 2 or combined_score >= 0.2:
                sentiment = "Positive"
            elif combined_score >= 0.05:
                sentiment = "Positive"
            else:
                sentiment = "Neutral"
        else:
            # Use combined score with tighter thresholds
            if combined_score >= 0.15:
                sentiment = "Positive"
            elif combined_score <= -0.15:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

        # Final validation - check for obvious patterns
        if any(phrase in normalized_comment for phrase in ['i love', 'so good', 'very good', 'really good']):
            sentiment = "Positive"
        elif any(
                phrase in normalized_comment for phrase in ['i hate', 'so bad', 'very bad', 'really bad', 'fuck this']):
            sentiment = "Negative"

        sentiments.append((sentiment, cleaned_comment))

    return sentiments


# ------------------- Fetching YouTube Comments -------------------
def fetch_comments(api_key, video_id, max_results=200):
    """Fetch YouTube comments with specified limit"""
    comments = []
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

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
            # Filter out very short or empty comments
            if len(comment.strip()) > 5:
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
    return comments[:max_results]  # Ensure we don't exceed the limit


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


# --------------- bar chart---------------
def plot_3d_bar_chart(sentiment_counts, analysis_id):
    # Clear any existing plots to prevent memory issues
    plt.clf()
    plt.close('all')
    # Map each sentiment to a specific color
    color_map = {
        "Positive": "#2196F3",  # Blue
        "Negative": "#F44336",  # Red
        "Neutral": "#4CAF50"  # Green
    }

    labels = sorted(sentiment_counts.keys())
    values = [sentiment_counts[lbl] for lbl in labels]
    total = sum(values) if sum(values) > 0 else 1
    percentages = [v / total * 100 for v in values]
    colors = [color_map[lbl] for lbl in labels]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    xpos = np.arange(len(labels))
    ypos = np.zeros_like(xpos)
    zpos = np.zeros_like(xpos)

    dx = np.ones_like(xpos) * 0.5
    dy = np.ones_like(xpos) * 0.5
    dz = values

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.9)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sentiments")
    ax.set_zlabel("Number of Comments")
    ax.set_title("Sentiment Analysis (YouTube Comments)")

    # Legend with numbers + percentages
    patches = []
    for lbl, val, pct, col in zip(labels, values, percentages, colors):
        patches.append(mpatches.Patch(color=col, label=f"{lbl}: {val} ({pct:.1f}%)"))

    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(-0.35, 1.1))

    plt.savefig(f"static/bar_chart_{analysis_id}.png", bbox_inches="tight", dpi=100)  # save in static folder
    plt.close(fig)
    plt.clf()  # Clear the current figure
    plt.close('all')  # Close all figures


# ------------------- Word Cloud -------------------
def generate_wordcloud(comments, analysis_id):
    # Clear any existing plots to prevent memory issues
    plt.clf()
    plt.close('all')
    text = " ".join(comments)
    wordcloud = WordCloud(
        width=1000, height=600, background_color="white",
        colormap="tab10", collocations=False, max_words=200
    ).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"static/wordcloud_{analysis_id}.png", dpi=100, bbox_inches='tight')
    plt.close()
    plt.clf()  # Clear the current figure
    plt.close('all')  # Close all figures


# ------------------- Flask Routes -------------------
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/history')
def history():
    """Display analysis history"""
    history_data = load_history()
    return render_template("index.html", history=history_data)


@app.route('/delete_history/<analysis_id>', methods=['POST'])
def delete_history_entry(analysis_id):
    """Delete a specific history entry"""
    from flask import jsonify

    if delete_from_history(analysis_id):
        return jsonify({'success': True, 'message': 'Analysis deleted successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to delete analysis'})


@app.route('/clear_all_history', methods=['POST'])
def clear_history():
    """Clear all history"""
    from flask import jsonify

    if clear_all_history():
        return jsonify({'success': True, 'message': 'All history cleared successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to clear history'})


@app.route('/analyze', methods=['POST'])
def analyze():
    api_key = "AIzaSyAHTiKNX-0vjDgeh4hFakvPrFXwUe5LsfU"
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
        return render_template("index.html", error="Invalid YouTube URL. Please use a valid YouTube URL.")

    try:
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add processing delay for preloader (easily adjustable)
        time.sleep(2)  # Initial delay

        comments = fetch_comments(api_key, video_id, max_results=max_comments)

        # Processing delay
        time.sleep(1)
        sentiments = analyze_sentiments_enhanced(comments)
        sentiment_counts = Counter([s for s, c in sentiments])

        # Analysis delay
        time.sleep(1.5)

        # Generate wordcloud and barchart with unique IDs
        generate_wordcloud(comments, analysis_id)

        # Visualization delay
        time.sleep(1)

        plot_3d_bar_chart(sentiment_counts, analysis_id)

        # Force garbage collection to free memory
        gc.collect()

        # Final processing delay
        time.sleep(0.5)

        # Save to history
        save_to_history(video_url, video_id, sentiment_counts, len(comments), analysis_id, timestamp,
                        max_comments_param)

        print(f"▶ Fetching comments for: {video_id}")
        print(f"Total comments fetched: {len(comments)}")
        print(f"Sentiment counts: {sentiment_counts}")
        print(f"Analysis ID: {analysis_id}")
        print("✅ API request executed and saved to history")

        return render_template(
            "index.html",
            analyzed=True,
            video_url=video_url,
            sentiments=sentiments,
            sentiment_counts=sentiment_counts,
            analysis_id=analysis_id,
            max_comments_requested=max_comments_param
        )
    except Exception as e:
        print(f"Error: {e}")
        return render_template("index.html", error=f"Error analyzing video: {str(e)}")


if __name__ == "__main__":
    nltk.download("vader_lexicon")
    app.run(debug=True)
