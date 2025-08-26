"""
Flask server for emotion detection application.

This module sets up a Flask web server that serves a UI for inputting text
and detecting emotions using the EmotionDetection package.
"""

from flask import Flask, request, render_template
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__,
           template_folder="oaqjp-final-project-emb-ai/templates",
           static_folder="oaqjp-final-project-emb-ai/static")

@app.route("/")
def index():
    """
    Renders the main index page.

    Returns:
        str: Rendered HTML template for index.html.
    """
    return render_template("index.html")

@app.route("/emotionDetector", methods=["GET", "POST"])
def detect_emotion():
    """
    Detects emotion from input text via GET or POST request.

    Handles both GET (query param textToAnalyze) and POST (JSON body with 'text').

    Returns:
        str: Formatted emotion scores or error message.
    """
    if request.method == "POST":
        data = request.get_json()
        text = data.get("text", "")
    else:  # GET
        text = request.args.get("textToAnalyze", "")

    result = emotion_detector(text)
    if result["dominant_emotion"] is None:
        return "Invalid text! Please try again!"

    response = (
        "For the given statement, the system response is "
        f"'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, "
        f"'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
