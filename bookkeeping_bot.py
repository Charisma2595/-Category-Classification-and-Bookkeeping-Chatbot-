import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
import chromadb

logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY not found in environment variables")

class BookkeepingBot:
    def __init__(self, csv_path: str = "dataset/bookkeeping_transactions.csv"):
        logger.info(f"Initializing BookkeepingBot with CSV: {csv_path}")
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            self.df = pd.read_csv(csv_path)
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df['month_name'] = self.df['date'].dt.strftime('%B').str.lower()
            logger.info(f"Successfully loaded {len(self.df)} transactions from CSV")
            
            self.setup_vector_store()
            self.setup_llm()
        except Exception as e:
            logger.error(f"Failed to initialize BookkeepingBot: {e}", exc_info=True)
            raise

    def setup_vector_store(self):
        try:
            logger.info("Setting up vector store")
            documents = self._create_documents()
            logger.info(f"Created {len(documents)} documents")
            
            
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Embeddings initialized")
            
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Create vector store
            logger.info("Creating Chroma vector store")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=chroma_client
            )
            logger.info("Vector store setup complete")
            
        except Exception as e:
            logger.error(f"Failed to set up vector store: {e}", exc_info=True)
            raise

    def _create_documents(self):
        try:
            logger.info("Creating documents from transaction data")
            documents = []
            # Process all transactions but in smaller batches
            batch_size = 10
            for i in range(0, len(self.df), batch_size):
                batch = self.df.iloc[i:i+batch_size]
                for _, row in batch.iterrows():
                    content = f"Transaction: {row['description']} for ${row['amount']} categorized as {row['category']} on {row['date']}"
                    metadata = {
                        "description": row["description"],
                        "amount": float(row["amount"]),
                        "category": row["category"],
                        "month": row["month_name"]
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(self.df) + batch_size - 1)//batch_size}")
            logger.info(f"Created {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Failed to create documents: {e}", exc_info=True)
            raise

    def setup_llm(self):
        try:
            logger.info("Setting up LLM chain")
            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name="llama3-70b-8192"
            )
            logger.info("ChatGroq initialized")

            system_prompt = PromptTemplate.from_template(
                """You are a bookkeeping assistant. Use the following transaction data to answer the question:
                {context}
                Focus on:
                - Uncategorized spend (category 'Unclassified')
                - Expense counts by category (e.g., Travel, Utilities)
                - Top vendors by total amount
                - key data information
                Be specific with numbers and months when mentioned. If data is missing, say so.
                Question: {question}
                Answer:"""
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type="stuff",
                verbose=True,
                return_source_documents=True
            )
            self.qa_chain.combine_docs_chain.llm_chain.prompt = system_prompt
            logger.info("LLM chain setup complete")
        except Exception as e:
            logger.error(f"Failed to set up LLM chain: {e}", exc_info=True)
            raise

    def get_answer(self, query: str) -> str:
        if not query:
            logger.warning("Empty query received")
            return "Please provide a valid question."
            
        try:
            logger.info(f"Processing query: {query}")
            result = self.qa_chain.invoke({
                "question": query,
                "chat_history": []
            })
            logger.info(f"Retrieved {len(result['source_documents'])} documents")
            answer = result["answer"]
            logger.info(f"Generated answer: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return f"Error processing query: {str(e)}"