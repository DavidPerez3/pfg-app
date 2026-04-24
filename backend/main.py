from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from agent import graph

app = FastAPI(title="PFG App Backend")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    # In case the frontend sends more context we can add it here
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    messages: List[ChatMessage]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "PFG App Backend running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Convert Pydantic request to the format expected by LangGraph
        input_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # We invoke the graph with the current messages
        # Ideally, LangGraph keeps state, but here we just pass the full history to the graph
        config = {"configurable": {"thread_id": request.thread_id or "default_thread"}}
        
        # Invoke graph
        result = graph.invoke({"messages": input_messages}, config=config)
        
        # The result["messages"] will contain the updated messages
        # Depending on how the state graph merges messages, we only want the newly added ones
        # or the full sequence. Langchain models return specific types of AIMessage etc.
        # But we are using basic dicts in our dummy node
        
        response_messages = [{"role": msg["role"], "content": msg["content"]} for msg in result["messages"]]
        
        return ChatResponse(messages=response_messages)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
