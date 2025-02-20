from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load pre-trained model
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        news_text = request.form["news_text"]
        news_vector = vectorizer.transform([news_text])
        result = "Fake" if model.predict(news_vector)[0] == 1 else "Real"
    return render_template("main.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
