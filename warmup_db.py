import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

# Load environment variables from .env file
load_dotenv()

PERSIST_DIRECTORY = "chroma_db"

chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', ''))

# Create or get the collection
collection = chroma_client.get_or_create_collection(
    name="room_embeddings",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY', ''),
        model_name="text-embedding-3-small"
    )
)

def generate_summary(text):
    """Generate a summary using OpenAI's API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
                {"role": "user", "content": f"Please provide a brief summary of this text in persian language: {text}"}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

def create_embedding(text):
    """Create an embedding using OpenAI's API."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

def query_similar_rooms(query_text, n_results=5):
    """Query ChromaDB for similar rooms based on the query text."""
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return None

def process_room_details():
    """Process room details, generate summaries, and create embeddings."""
    # Load the room details
    try:
        with open('room_details.json', 'r', encoding='utf-8') as f:
            room_details = json.load(f)
    except Exception as e:
        print(f"Error loading room_details.json: {e}")
        return

    # Initialize the output structure
    processed_data = {
        "items": []
    }

    # Process each item
    for idx, item in enumerate(tqdm(room_details[:5], desc="Processing items")):
        summary = generate_summary(str(item))

        print(summary)
        
        if summary:
            # Extract metadata
            metadata = {
                "min_price": str(item.get('min_price', 'N/A')),
                "extra_price": str(item.get('extra_price', 'N/A')),
                "city": str(item.get('city', {}).get('name', 'N/A')),
                "title": str(item.get('title', 'N/A')),
                "description": str(item.get('description', 'N/A')),
                "id": str(item.get('id', str(idx))),
                "url": str(item.get('url', 'N/A')),
            }
            
            # Add to ChromaDB
            collection.add(
                documents=[summary],
                metadatas=[metadata],
                ids=[f"room_{idx}"]
            )
            
            # Store in processed data
            processed_item = {
                "original_item": item,
                "summary": summary,
                "metadata": metadata
            }
            processed_data["items"].append(processed_item)
        
        time.sleep(0.5)

    # Query for similar rooms
    query_text = "کلبه کاهگلی میوه منظره جنگلی"
    results = query_similar_rooms(query_text)
    
    print('------------------------------------------')
    print("Query Results:")
    if results:
        for i, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
            print(f"\nResult {i+1}:")
            print(f"Summary: {doc}")
            print(f"Metadata: {metadata}")
            print(f"Similarity Score: {1 - distance}")  # Convert distance to similarity score
    print('------------------------------------------')

    # Save the processed data
    try:
        with open('processed_room_details.json', 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print("Successfully saved processed data to processed_room_details.json")
    except Exception as e:
        print(f"Error saving processed data: {e}")

if __name__ == "__main__":
    process_room_details()
