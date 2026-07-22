"""
Viral Post Predictor — ML Core
Feature engineering + XGBoost classifier + SHAP explanations
Platform-aware scoring for Twitter, Reddit, Instagram
"""

import re
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import classification_report
import shap
import pickle
import os

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

# ── Power words known to boost engagement ─────────────────────────────────
POWER_WORDS = [
    "you", "your", "free", "secret", "proven", "finally", "never", "always",
    "instantly", "guaranteed", "discover", "exclusive", "urgent", "limited",
    "shocking", "surprising", "incredible", "amazing", "breakthrough",
    "warning", "stop", "important", "announcing", "new", "now", "today",
    "just", "only", "imagine", "because", "easy", "simple", "truth",
    "hack", "mistake", "actually", "nobody", "everyone", "realize",
    "honest", "real", "admit", "confession", "unpopular", "change",
]

QUESTION_STARTERS = ["what", "why", "how", "when", "which", "who", "would",
                     "should", "could", "do you", "did you", "have you",
                     "is this", "are you", "can you", "will you", "have you"]

HOOK_PHRASES = [
    "thread", "🧵", "here's why", "hot take", "unpopular opinion",
    "nobody talks", "stop doing", "i realized", "i learned", "after",
    "years ago", "changed my", "truth about", "mistake i made",
    "what i wish", "honest", "real talk",
]

analyzer = SentimentIntensityAnalyzer()


def extract_features(text: str, platform: str = "twitter") -> dict:
    """Extract all hand-engineered virality features from post text."""
    text = text.strip()
    text_lower = text.lower()
    words = text_lower.split()

    # ── Length features ───────────────────────────────────────────────────
    char_len = len(text)
    word_count = len(words)

    # ── Sentiment ─────────────────────────────────────────────────────────
    scores = analyzer.polarity_scores(text)
    sentiment_compound = scores["compound"]
    sentiment_strength = abs(sentiment_compound)
    is_positive = int(sentiment_compound > 0.2)
    is_negative = int(sentiment_compound < -0.2)

    # ── Structural signals ────────────────────────────────────────────────
    has_question = int(
        text.strip().endswith("?") or
        any(text_lower.startswith(q) for q in QUESTION_STARTERS) or
        text_lower.count("?") >= 1
    )
    exclamation_count = min(text.count("!"), 3)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    # ── Power words ───────────────────────────────────────────────────────
    power_word_count = sum(1 for w in POWER_WORDS if w in text_lower)

    # ── Hook phrases (strong signal for virality) ─────────────────────────
    hook_count = sum(1 for h in HOOK_PHRASES if h in text_lower)
    has_hook = int(hook_count > 0)

    # ── Emoji signals ─────────────────────────────────────────────────────
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff]|"
        "[\U0001F300-\U0001F9FF]|"
        "[\U00002702-\U000027B0]",
        flags=re.UNICODE
    )
    emojis = emoji_pattern.findall(text)
    emoji_count = len(emojis)
    has_emoji = int(emoji_count > 0)

    # Platform-specific emoji sweet spots
    if platform == "twitter":
        emoji_sweet_spot = int(1 <= emoji_count <= 3)
    elif platform == "instagram":
        emoji_sweet_spot = int(1 <= emoji_count <= 5)
    else:  # reddit
        emoji_sweet_spot = int(emoji_count == 0)  # Reddit dislikes emojis

    # ── Hashtag signals ───────────────────────────────────────────────────
    hashtags = re.findall(r"#\w+", text)
    hashtag_count = len(hashtags)
    mention_count = len(re.findall(r"@\w+", text))

    # Platform-specific hashtag sweet spots
    if platform == "twitter":
        hashtag_sweet_spot = int(1 <= hashtag_count <= 2)
    elif platform == "instagram":
        hashtag_sweet_spot = int(3 <= hashtag_count <= 10)
    else:  # reddit — hashtags irrelevant
        hashtag_sweet_spot = int(hashtag_count == 0)

    # Too many hashtags is always bad (spam signal)
    hashtag_spam = int(
        (platform == "twitter" and hashtag_count > 3) or
        (platform == "instagram" and hashtag_count > 15) or
        (platform == "reddit" and hashtag_count > 1)
    )

    # ── Readability ───────────────────────────────────────────────────────
    try:
        readability = textstat.flesch_reading_ease(text)
    except Exception:
        readability = 50.0
    is_readable = int(readability > 55)

    # ── URL presence ──────────────────────────────────────────────────────
    has_url = int(bool(re.search(r"https?://\S+", text)))

    # ── Platform-specific length sweet spot ───────────────────────────────
    if platform == "twitter":
        length_sweet_spot = int(60 <= char_len <= 220)
        length_too_long = int(char_len > 250)
    elif platform == "instagram":
        length_sweet_spot = int(80 <= char_len <= 300)
        length_too_long = int(char_len > 400)
    else:  # reddit
        length_sweet_spot = int(40 <= char_len <= 120)
        length_too_long = int(char_len > 150)

    # ── Number presence (specificity = credibility) ───────────────────────
    has_number = int(bool(re.search(r"\b\d+\b", text)))

    # ── Storytelling signals ──────────────────────────────────────────────
    first_person = int(bool(re.search(r"\b(i |i'|my |me |myself)\b", text_lower)))
    has_contrast = int(bool(re.search(
        r"\b(but|however|yet|although|despite|instead|rather)\b", text_lower
    )))

    # ── Call to action ────────────────────────────────────────────────────
    has_cta = int(bool(re.search(
        r"\b(comment|share|follow|like|retweet|rt|tag|dm|save|link in bio|check)\b",
        text_lower
    )))

    return {
        "char_len": char_len,
        "word_count": word_count,
        "sentiment_compound": sentiment_compound,
        "sentiment_strength": sentiment_strength,
        "is_positive": is_positive,
        "is_negative": is_negative,
        "has_question": has_question,
        "exclamation_count": exclamation_count,
        "caps_ratio": caps_ratio,
        "power_word_count": power_word_count,
        "has_hook": has_hook,
        "hook_count": hook_count,
        "emoji_count": emoji_count,
        "has_emoji": has_emoji,
        "emoji_sweet_spot": emoji_sweet_spot,
        "hashtag_count": hashtag_count,
        "hashtag_sweet_spot": hashtag_sweet_spot,
        "hashtag_spam": hashtag_spam,
        "mention_count": mention_count,
        "readability": readability,
        "is_readable": is_readable,
        "has_url": has_url,
        "length_sweet_spot": length_sweet_spot,
        "length_too_long": length_too_long,
        "has_number": has_number,
        "first_person": first_person,
        "has_contrast": has_contrast,
        "has_cta": has_cta,
        # Platform one-hot — without this the model has no way to know that
        # e.g. a high hashtag_count means something different on Instagram
        # than on Twitter or Reddit, so it ends up learning a single global
        # (and often wrong) rule from whichever platform dominates the signal.
        "platform_twitter": int(platform == "twitter"),
        "platform_reddit": int(platform == "reddit"),
        "platform_instagram": int(platform == "instagram"),
    }


# ── Platform-specific training data ───────────────────────────────────────

TWITTER_VIRAL = [
    "I quit my job 2 years ago to build startups. Here's everything I wish someone told me 🧵",
    "Nobody talks about this but burnout is destroying an entire generation of developers.",
    "Hot take: Most coding tutorials teach you the wrong things. Here's what actually matters:",
    "After 100+ job rejections, I finally got hired at Google. The one thing that changed everything:",
    "Stop using to-do lists. I switched to this system 6 months ago and my productivity 3x'd.",
    "I analyzed 500 viral tweets. Here's the exact formula they all share 🧵",
    "Unpopular opinion: Your side hustle is probably keeping you broke. Here's why:",
    "I built a SaaS in 30 days that now makes $3k/month. Full breakdown 👇",
    "What I learned working at 5 different startups that no one tells you in interviews:",
    "The reason you're not getting callbacks has nothing to do with your skills.",
    "3 years of learning to code. This is the honest truth nobody posts about.",
    "I made every mistake possible in my 20s. Here's what I'd tell my younger self 🧵",
    "Why do seniors earn 3x more than juniors doing the same work? Thread on negotiation:",
    "Just shipped my first open source project. 200 stars in 24 hours. What worked:",
    "The coding advice that actually got me hired (not the stuff on YouTube):",
    "I've interviewed 300+ engineers. These are the mistakes that instantly disqualify people:",
    "Real talk: most 'passive income' content is selling you a lie. Here's what's actually passive:",
    "6 months ago I had $0 in savings. Here's the exact system that changed everything:",
    "This one Python trick saved me 3 hours every single week. Sharing it for free:",
    "My manager told me I'd never be a senior dev. 18 months later I proved them wrong. Story 👇",
]

TWITTER_FLOP = [
    "Working on a new project today using Python and Flask.",
    "Just finished reading a book on software architecture.",
    "Attended a meetup about cloud computing this evening.",
    "Updated my LinkedIn profile with my new skills.",
    "Started learning a new framework this week.",
    "My pull request got merged today.",
    "Completed another module in my online course.",
    "Set up a new development environment on my laptop.",
    "Had a productive standup meeting this morning.",
    "Reviewed some code for a colleague today.",
    "Pushed some bug fixes to the repository.",
    "Installed a new VS Code extension I found.",
    "Read an interesting blog post about databases.",
    "Configured my terminal with a new theme.",
    "Downloaded the latest version of Node.js.",
]

REDDIT_VIRAL = [
    "I figured out why my code was broken after 6 hours. It was a missing semicolon.",
    "What's the one thing you wish you knew before starting your CS degree?",
    "Genuine question: is a CS degree still worth it in 2024?",
    "I got rejected from 47 companies before landing my first dev job. AMA.",
    "Why does every senior dev I know say 'it depends' to literally everything?",
    "Am I the only one who finds LeetCode completely useless for actual work?",
    "I just discovered you can do this in Python and my mind is blown.",
    "After 10 years in tech, here's my honest take on work-life balance.",
    "What killed my productivity wasn't social media. It was meetings.",
    "The interview question that stumped me even though I use it every day.",
    "Hot take: most design patterns are overcomplicated solutions to simple problems.",
    "I passed the Google interview and turned down the offer. Here's why.",
    "The one book that changed how I think about writing code.",
    "Why I quit a $200k job and what happened after.",
    "What actually happens inside a for loop that nobody explains properly.",
]

REDDIT_FLOP = [
    "Here is my new project built with React and Node.",
    "I made a to-do app for practice.",
    "Just completed a tutorial on machine learning.",
    "Sharing my first GitHub repository.",
    "Learning Python as my first language.",
    "I made a weather app using an API.",
    "Built a calculator in JavaScript.",
    "My portfolio website is now live.",
    "Finished a Udemy course today.",
    "Created a simple CRUD application.",
]

INSTAGRAM_VIRAL = [
    "Not lucky. Just obsessed 🔥 3 years of failing, learning, and refusing to quit. What's your story? 👇",
    "Every expert was once a beginner who refused to give up ✨ Save this for when you feel like quitting 💫",
    "The version of you 5 years from now is watching. Make them proud 🚀 #growth #mindset",
    "Real ones know: the hardest part isn't starting. It's starting again after you fail 💪 Tag someone who needs this",
    "Plot twist: your 'failures' are actually data points 📊 Keep going. The algorithm of life rewards persistence ✨",
    "Day 1 vs Day 365 of learning to code 💻 The progress isn't always visible but it's always happening 🌱",
    "Nobody posts the 2am debugging sessions. The 47 rejected applications. The self-doubt. I'm posting it. 💙",
    "Reminder: you don't need permission to start. You just need to start 🔥 What are you waiting for?",
    "3 things that actually helped me land my dream job (that nobody talks about) 👇 Save this!",
    "Hot take: consistency > motivation. Motivation is a feeling. Consistency is a decision 💯",
    "The gap between where you are and where you want to be is called: showing up every single day 🌟",
    "I was told I wasn't technical enough. Now I lead a team of engineers 👩‍💻 Doubt me more.",
    "Your LinkedIn is not your worth. Your GitHub is not your worth. YOU are your worth 💙 #developerlife",
    "6am workout ✅ Deep work session ✅ Shipped a feature ✅ What did you build today? 👇",
    "The best investment I ever made wasn't in stocks. It was in my own skills 📈 What are you learning?",
]

INSTAGRAM_FLOP = [
    "Working on some new projects today. Feeling productive! #coding #tech #developer #python #ML #programming",
    "Just finished my daily workout and now heading to work. Good morning everyone! #morning #motivation #lifestyle",
    "Updated my portfolio website today. Check it out! #webdev #portfolio #developer",
    "Learning something new every day. Today it was React hooks. #react #javascript #coding",
    "Had a great day at the office. Team lunch was amazing! #work #team #lunch",
    "Reading a great book on productivity this weekend. Highly recommend! #books #reading #productivity",
    "New blog post is up! Link in bio. #blog #tech #writing",
    "Coffee and code. The perfect morning combo ☕ #coding #coffee #developer",
    "Just hit 100 followers! Thank you all so much! #grateful #milestone",
    "Sharing my workspace setup. What do you think? #setup #workspace #productivity",
]


def generate_platform_data(platform: str, viral_posts: list, flop_posts: list,
                            n_per_class: int = 500) -> list:
    """Generate labeled records for a specific platform.
    Caller is responsible for seeding (see generate_training_data) so that
    augmentation noise isn't accidentally correlated across platforms.

    Each record carries a "group" id identifying which base example it was
    augmented from (e.g. "twitter_viral_3"). There are only ~15-20 unique
    base sentences per class per platform, repeated/lightly-perturbed to
    fill out n_per_class — so a *random* train/test split would put several
    near-duplicates of the same sentence on both sides, letting the model
    "pass the test" by memorization rather than generalization. The group id
    lets train_model() split by base example instead, so held-out accuracy
    reflects performance on genuinely unseen writing.
    """
    records = []

    # Viral posts with augmentation
    for i in range(n_per_class):
        base_idx = i % len(viral_posts)
        text = viral_posts[base_idx]
        # Slight variations to augment
        if i >= len(viral_posts):
            words = text.split()
            if len(words) > 5 and np.random.random() > 0.5:
                idx = np.random.randint(1, len(words) - 1)
                words.insert(idx, np.random.choice(["really", "honestly", "actually", "genuinely"]))
                text = " ".join(words)
        try:
            feats = extract_features(text, platform)
            feats["label"] = 1
            feats["text"] = text
            feats["group"] = f"{platform}_viral_{base_idx}"
            records.append(feats)
        except Exception:
            continue

    # Flop posts with augmentation
    for i in range(n_per_class):
        base_idx = i % len(flop_posts)
        text = flop_posts[base_idx]
        if i >= len(flop_posts):
            text = text + f" #{np.random.choice(['coding', 'tech', 'dev', 'python', 'ml'])}" * np.random.randint(1, 4)
        try:
            feats = extract_features(text, platform)
            feats["label"] = 0
            feats["text"] = text
            feats["group"] = f"{platform}_flop_{base_idx}"
            records.append(feats)
        except Exception:
            continue

    return records


def load_real_data(n: int = 2000) -> list:
    """Load and label real tweets from CSV if available."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tweets.csv")

    if not os.path.exists(data_path):
        print("No real dataset found — using curated training data only.")
        return []

    print(f"Loading real dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path, low_memory=False)
        df.columns = df.columns.str.strip()

        if "RetweetCount" not in df.columns or "text" not in df.columns:
            print("Required columns not found in CSV.")
            return []

        # Combined engagement score: retweets weighted more than likes
        df["RetweetCount"] = pd.to_numeric(df["RetweetCount"], errors="coerce").fillna(0)
        if "Likes" in df.columns:
            df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0)
            df["engagement"] = df["RetweetCount"] * 3 + df["Likes"]
        else:
            df["engagement"] = df["RetweetCount"]

        df = df[["text", "engagement"]].dropna()
        df = df[df["text"].str.strip().str.len() > 20]
        df = df[df["engagement"] >= 0]

        # Use top 20% as viral — stricter and more meaningful
        threshold = df["engagement"].quantile(0.80)
        # Ensure threshold is meaningful
        if threshold < 3:
            threshold = 3
        df["label"] = (df["engagement"] >= threshold).astype(int)

        print(f"Real data: {len(df)} rows | Viral threshold: {threshold:.0f} | "
              f"Balance: {df['label'].value_counts().to_dict()}")

        # Sample balanced classes
        viral_df = df[df["label"] == 1].sample(min(n // 2, df["label"].sum()), random_state=42)
        flop_df  = df[df["label"] == 0].sample(min(n // 2, (df["label"] == 0).sum()), random_state=42)
        df_balanced = pd.concat([viral_df, flop_df])

        records = []
        for row_idx, row in df_balanced.iterrows():
            try:
                feats = extract_features(str(row["text"]), "twitter")
                feats["label"] = int(row["label"])
                feats["text"] = row["text"]
                feats["group"] = f"real_{row_idx}"  # each real tweet is its own group
                records.append(feats)
            except Exception:
                continue

        print(f"Real data records extracted: {len(records)}")
        return records

    except Exception as e:
        print(f"Error loading real data: {e}")
        return []


def generate_training_data() -> pd.DataFrame:
    """Build full training set: curated platform data + real CSV data."""
    records = []
    np.random.seed(42)  # seed once, here, so the three platform calls below
                         # don't each restart from the same random sequence

    print("Building platform-aware training data...")

    # Twitter data
    tw = generate_platform_data("twitter", TWITTER_VIRAL, TWITTER_FLOP, n_per_class=600)
    records.extend(tw)
    print(f"  Twitter: {len(tw)} records")

    # Reddit data
    rd = generate_platform_data("reddit", REDDIT_VIRAL, REDDIT_FLOP, n_per_class=400)
    records.extend(rd)
    print(f"  Reddit:  {len(rd)} records")

    # Instagram data
    ig = generate_platform_data("instagram", INSTAGRAM_VIRAL, INSTAGRAM_FLOP, n_per_class=500)
    records.extend(ig)
    print(f"  Instagram: {len(ig)} records")

    # Real CSV data (Twitter-style)
    real = load_real_data(n=1500)
    records.extend(real)
    print(f"  Real CSV: {len(real)} records")

    df = pd.DataFrame(records)
    print(f"Total training records: {len(df)} | "
          f"Viral: {df['label'].sum()} | Not viral: {(df['label']==0).sum()}")
    return df


FEATURE_COLS = [
    "char_len", "word_count", "sentiment_compound", "sentiment_strength",
    "is_positive", "is_negative", "has_question", "exclamation_count",
    "caps_ratio", "power_word_count", "has_hook", "hook_count",
    "emoji_count", "has_emoji", "emoji_sweet_spot",
    "hashtag_count", "hashtag_sweet_spot", "hashtag_spam", "mention_count",
    "readability", "is_readable", "has_url",
    "length_sweet_spot", "length_too_long",
    "has_number", "first_person", "has_contrast", "has_cta",
]

FEATURE_LABELS = {
    "sentiment_strength": "Emotional intensity",
    "power_word_count":   "Power words",
    "has_hook":           "Hook phrase",
    "has_question":       "Question format",
    "length_sweet_spot":  "Optimal length",
    "emoji_sweet_spot":   "Emoji usage",
    "hashtag_sweet_spot": "Hashtag count",
    "hashtag_spam":       "Hashtag spam",
    "is_readable":        "Readability",
    "has_number":         "Specificity (numbers)",
    "caps_ratio":         "Capitalisation",
    "exclamation_count":  "Exclamation marks",
    "has_url":            "Contains URL",
    "sentiment_compound": "Sentiment",
    "first_person":       "Personal story (I/my)",
    "has_contrast":       "Contrast words (but/yet)",
    "has_cta":            "Call to action",
    "length_too_long":    "Too long",
    "is_positive":        "Positive tone",
    "is_negative":        "Negative/urgent tone",
}


def _build_classifier(early_stopping: bool) -> XGBClassifier:
    """Single source of truth for model hyperparameters, so the CV pass and
    the final fit always use the identical (regularized) configuration.

    Compared to the original config (max_depth=6, no regularization, no
    early stopping), this is deliberately shallower and more penalized:
    the previous settings reached 100%/100% train/test accuracy, which is
    a classic overfitting signature on a synthetic dataset this small —
    the model was memorizing exact training examples rather than learning
    generalizable virality signals, which is why held-out rewrites could
    score wildly differently from run to run.
    """
    kwargs = dict(
        n_estimators=300,
        max_depth=4,              # was 6 — shallower trees generalize better
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=6,       # was 3 — require more evidence per split
        gamma=0.2,                # was 0.1 — higher min loss-reduction to split
        reg_alpha=0.5,            # L1 regularization (new)
        reg_lambda=2.0,           # L2 regularization (new)
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,                 # forces deterministic tree building —
                                   # XGBoost's multi-threaded histogram method
                                   # can produce different splits on different
                                   # runs even with a fixed random_state
    )
    if early_stopping:
        kwargs["early_stopping_rounds"] = 20
    return XGBClassifier(**kwargs)


def train_model():
    print("Generating training data...")
    df = generate_training_data()

    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = df["group"].values

    # ── Honest generalization estimate ──────────────────────────────────
    # Each base sentence is augmented into ~20-40 near-duplicate records
    # (see generate_platform_data). A random/stratified split ignores that
    # and happily puts near-duplicates of the same sentence on both sides
    # of train/test — which is exactly how the old code reported 100%/100%
    # accuracy: the model wasn't being tested on anything it hadn't already
    # half-seen. GroupKFold instead keeps every augmentation of a given
    # base sentence together on one side, so the reported score reflects
    # performance on genuinely unseen writing.
    print("\nRunning 5-fold grouped cross-validation (grouped by base example)...")
    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    cv = GroupKFold(n_splits=n_splits)
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X[train_idx])
        X_val = fold_scaler.transform(X[val_idx])
        fold_model = _build_classifier(early_stopping=False)
        fold_model.fit(X_tr, y[train_idx])
        fold_acc = fold_model.score(X_val, y[val_idx])
        cv_scores.append(fold_acc)
        print(f"  Fold {fold}: accuracy = {fold_acc:.3f}")
    cv_scores = np.array(cv_scores)
    print(f"Grouped cross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("(This is the number to report/trust — it's evaluated on base")
    print(" sentences the model never saw any augmented copy of.)")

    # ── Final model, trained on a held-out split with early stopping ────
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\nTraining final XGBoost model...")
    model = _build_classifier(early_stopping=True)
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )
    print(f"Stopped at boosting round: {model.best_iteration}")

    print("\nHeld-out test performance (grouped split, unseen base sentences):")
    preds = model.predict(X_test_s)
    print(classification_report(y_test, preds, target_names=["Not Viral", "Viral"]))

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"Saved model.pkl")
    return model, scaler


def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    if not os.path.exists(model_path):
        return train_model()
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]


def predict(text: str, platform: str, model, scaler):
    """Return virality score 0-100 + per-feature breakdown."""
    feats = extract_features(text, platform)
    X = np.array([[feats[c] for c in FEATURE_COLS]])
    X_s = scaler.transform(X)

    prob = model.predict_proba(X_s)[0][1]
    score = int(prob * 100)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_s)[0]

    breakdown = []
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith("platform_"):
            continue  # always constant for a given post — not a useful "signal" to show
        label = FEATURE_LABELS.get(col, col)
        impact = float(shap_vals[i])
        value  = feats[col]
        breakdown.append({
            "feature": col,
            "label":   label,
            "impact":  impact,
            "value":   value,
        })

    breakdown.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return score, breakdown, feats


def get_weaknesses(feats: dict, score: int, platform: str) -> list[str]:
    """Return specific, platform-aware rewrite instructions."""
    issues = []

    # Hook check — most important for virality
    if not feats["has_hook"] and score < 70:
        issues.append("Add a hook phrase in the first line: 'nobody talks about this', 'hot take:', 'after X years', or end with 🧵")

    # Sentiment
    if feats["sentiment_strength"] < 0.2:
        issues.append("The tone is too neutral — add stronger emotion (excitement, surprise, urgency, or personal vulnerability)")

    # Platform-specific length
    if platform == "twitter":
        if feats["char_len"] > 250:
            issues.append(f"Too long for Twitter ({feats['char_len']} chars) — trim to under 220 characters")
        elif feats["char_len"] < 60:
            issues.append("Too short — expand to at least 60 characters with more context or a hook")
    elif platform == "instagram":
        if feats["char_len"] < 80:
            issues.append("Too short for Instagram — add a personal story or call to action to reach 100+ characters")
        if feats["hashtag_count"] < 3:
            issues.append(f"Add 5–10 relevant hashtags — Instagram rewards hashtag use (you only have {feats['hashtag_count']})")
        elif feats["hashtag_count"] > 15:
            issues.append(f"Too many hashtags ({feats['hashtag_count']}) — trim to 8–12 targeted ones")
    elif platform == "reddit":
        if feats["char_len"] > 150:
            issues.append(f"Too long for a Reddit title ({feats['char_len']} chars) — Reddit titles should be punchy and under 120 chars")
        if feats["hashtag_count"] > 0:
            issues.append("Remove all hashtags — Reddit users find hashtags annoying and they hurt upvotes")
        if feats["emoji_count"] > 1:
            issues.append("Remove emojis — Reddit culture prefers plain text titles")

    # Power words
    if feats["power_word_count"] == 0:
        issues.append("Add a power word: 'you', 'never', 'secret', 'finally', 'stop', 'honest', 'real talk', or 'nobody'")

    # Question format
    if not feats["has_question"] and platform in ("twitter", "reddit"):
        issues.append("Consider ending with a question to invite replies and boost engagement")

    # Personal story
    if not feats["first_person"] and score < 50:
        issues.append("Add a personal angle using 'I' — first-person posts consistently outperform generic advice")

    # Specificity
    if not feats["has_number"]:
        issues.append("Add a specific number for credibility (e.g. '3 years', '47 rejections', '10x faster')")

    # Hashtag spam on Twitter
    if platform == "twitter" and feats["hashtag_spam"]:
        issues.append(f"Way too many hashtags ({feats['hashtag_count']}) — cut to 1–2 max on Twitter, they look spammy")

    # CTA for Instagram
    if platform == "instagram" and not feats["has_cta"]:
        issues.append("Add a call to action: 'save this', 'tag someone', 'comment below', or 'follow for more'")

    return issues[:4]


if __name__ == "__main__":
    train_model()