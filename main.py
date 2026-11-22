from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <body style="background:#101c2c;color:white;text-align:center;font-family:Arial;padding:40px">
        <h1>ClefTalk</h1>
        <h2>Raspberry Pi Connected!</h2>
        <p>Your wired phone display is working.</p>
    </body>
    </html>
    """