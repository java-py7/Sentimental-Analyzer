from database import db
from datetime import datetime
from flask_login import UserMixin

# ------------------- User -------------------

class User(UserMixin, db.Model):

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    profile_picture = db.Column(db.String(500))

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    analyses = db.relationship(
        "Analysis",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )


# ------------------- Analysis -------------------

class Analysis(db.Model):

    __tablename__ = "analysis"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    video_title = db.Column(db.String(300))
    video_url = db.Column(db.String(500))
    video_id = db.Column(db.String(50))
    positive = db.Column(db.Integer)
    negative = db.Column(db.Integer)
    neutral = db.Column(db.Integer)
    total_comments = db.Column(db.Integer)
    chart_image = db.Column(db.LargeBinary)
    wordcloud_image = db.Column(db.LargeBinary)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    comments = db.relationship(
        "Comment",
        backref="analysis",
        lazy=True,
        cascade="all, delete-orphan"
    )


# ------------------- Comments -------------------

class Comment(db.Model):

    __tablename__ = "comments"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    analysis_id = db.Column(
        db.Integer,
        db.ForeignKey("analysis.id"),
        nullable=False,
        index=True
    )

    sentiment = db.Column(
        db.String(20),
        nullable=False
    )

    comment = db.Column(
        db.Text,
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

