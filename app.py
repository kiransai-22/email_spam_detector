
from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load the pre-trained spam detection model
model_path = os.path.join("model", "spam_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        email_text = request.form["email"]
        prediction = model.predict([email_text])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
