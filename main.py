from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from agent import get_query_from_agent
app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat")
async def chat(query: str):
    if not query:
        return {"status": "error", "message": "No query provided"}
    response = get_query_from_agent(query)
    return {"status": "success", "response": response}