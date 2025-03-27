from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from classify import classify  
import pandas as pd
import io
from bookkeeping_bot import BookkeepingBot
from pydantic import BaseModel
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

app = FastAPI(title="Transaction Classifier and Bookkeeping Chatbot API")

# Initialize bot globally with lazy loading
bot = None
try:
    bot = BookkeepingBot(csv_path="dataset/bookkeeping_transactions.csv")
    logger.info("BookkeepingBot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize BookkeepingBot: {e}", exc_info=True)

def process_csv(file_content: bytes):
    df = pd.read_csv(io.BytesIO(file_content))
    if "description" not in df.columns or "amount" not in df.columns:
        raise ValueError("CSV must contain 'description' and 'amount' columns")
    results = classify(list(zip(df["description"], df["amount"])))
    normalized_results = []
    for r in results:
        if isinstance(r, str):
            normalized_results.append({"category": r, "confidence": 1.0, "needs_review": False})
        elif isinstance(r, dict):
            normalized_results.append(r)
        else:
            normalized_results.append({"category": "Unclassified", "confidence": 0.0, "needs_review": True})
    df["category"] = [r["category"] for r in normalized_results]
    df["confidence"] = [r["confidence"] for r in normalized_results]
    df["needs_review"] = [r["needs_review"] for r in normalized_results]
    return df

@app.post("/classify-csv/")
async def classify_csv_endpoint(file: UploadFile):
    try:
        content = await file.read()
        df = process_csv(content)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=classified_transactions.csv"}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"CSV endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

class ChatQuery(BaseModel):
    query: str

@app.post("/chatbot/")
async def chatbot_endpoint(query_data: ChatQuery):
    if bot is None:
        logger.error("BookkeepingBot not initialized")
        raise HTTPException(status_code=503, detail="Chatbot service is not available due to initialization failure")
    
    try:
        logger.info(f"Processing query: {query_data.query}")
        response = bot.get_answer(query_data.query)
        logger.info(f"Generated response: {response}")
        return {"query": query_data.query, "response": response}
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    status = "Chatbot available" if bot else "Chatbot unavailable"
    return {"message": f"Welcome to the Transaction Classifier and Bookkeeping Chatbot API. {status}"}

if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)