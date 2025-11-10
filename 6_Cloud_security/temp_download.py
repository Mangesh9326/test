from flask import Flask, send_file, request, abort, make_response
from datetime import datetime, timedelta
import secrets
import os

app = Flask(__name__)
TOKENS = {} 

SAMPLE_FILE = "sample.txt"

@app.route("/generate_link")
def generate_link():
    token = secrets.token_urlsafe(16)
    expiry = datetime.now() + timedelta(seconds=60) 
    TOKENS[token] = expiry

    html = f"""
    <html>
      <head><meta charset="utf-8"><title>Temporary Link</title></head>
      <body>
        <p>Temporary link created. Use it within 60 seconds:</p>
        <p><a href="/download?token={token}">Download Link</a></p>
      </body>
    </html>
    """
    response = make_response(html)
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response

@app.route("/download")
def download():
    token = request.args.get("token")
    if not token:
        return make_response("Missing token.", 400)

    expiry = TOKENS.get(token)
    if not expiry:
        html = "<p>Invalid or expired link. Please generate a new one.</p>"
        return make_response(html, 403)

    if datetime.now() >= expiry:
        TOKENS.pop(token, None)
        html = "<p>Link expired. Please generate a new one.</p>"
        return make_response(html, 403)

    TOKENS.pop(token, None)

    if not os.path.exists(SAMPLE_FILE):
        return make_response("Requested file not found on server.", 404)

    return send_file(SAMPLE_FILE, as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists(SAMPLE_FILE):
        with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
            f.write("This is a sample file for download.\n")
    app.run(debug=True)
