from flask import Flask, render_template, request
from app.ai import calc_ypred

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/index", methods=["post"])
def post():
    ypred = calc_ypred(request.form)
    return render_template("index.html", ypred=ypred)


if __name__ == "__main__":
    app.run(debug=True)
