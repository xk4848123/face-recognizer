from flask import Flask, render_template, request

import worker

app = Flask(__name__)


@app.route("/add_face")
def add_face():
    return render_template("InputFace.html")


@app.route("/test_face")
def test_face():
    return render_template("testFace.html")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/face/init", methods=["GET", "POST"])
def init():
    return worker.init()


@app.route("/face/identify", methods=["POST"])
def identify():
    return worker.identify()


@app.route("/face/identify_base64", methods=["POST"])
def identify_base64():
    return worker.identify_base64()


@app.route("/face/add_face", methods=["POST"])
def uploadPic():
    return worker.uploadPic()


@app.route("/face/add_face_base64", methods=["POST"])
def uploadPicBase64():
    return worker.uploadPicBase64()


@app.route("/face/extractEigenvalue", methods=["POST"])
def extractEigenvalue():
    return worker.extractEigenvalue()


if __name__ == '__main__':
    app.run()
