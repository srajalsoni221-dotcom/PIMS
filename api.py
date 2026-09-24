# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
def login(request: LoginRequest):
    # Dummy verification. Replace this with real Database queries (e.g., PostgreSQL / SQLite)
    if request.username == "admin" and request.password == "pims123":
        return {"status": "success", "message": "Authenticated"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)