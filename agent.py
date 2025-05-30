from openai import OpenAI
from typing import List, Dict, Any
import json
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PERSIST_DIRECTORY = "chroma_db"
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
collection = chroma_client.get_or_create_collection(
    name="room_embeddings",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY', ''),
        model_name="text-embedding-3-small"
    )
)

def query_similar_rooms(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Tool to search for lodges and villas based on user query using ChromaDB.
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Transform the results into the expected format
        rooms = []
        for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            room = {
                "title": metadata.get('title', 'Unknown'),
                "type": "lodge" if "lodge" in doc.lower() else "villa",
                "description": doc,
                "price": metadata.get('min_price', 'Unknown'),
                "city": metadata.get('city', 'Unknown'),
                "rating": metadata.get('rating', 'Unknown'),
                "reviews_count": metadata.get('reviews_count', 'Unknown'),
                "image_url": metadata.get('image_url', 'Unknown'),
                "web_url": 'https://jajiga.com' + metadata.get('url', 'Unknown'),
                "similarity_score": 1 - distance  # Convert distance to similarity score
            }
            rooms.append(room)
        return rooms
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

class AccommodationAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_similar_rooms",
                    "description": "Search for lodges and villas based on user preferences",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query describing the desired accommodation"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def process_user_query(self, user_query: str) -> str:
        """
        Process the user's query and return a response using the OpenAI API
        """
        messages = [
            {"role": "system", "content": """
             You are a helpful assistant that helps users find lodges and villas and plan for their trip in Iran.
              TOOLS:
                - Use the query_similar_rooms function to search for accommodations.
              INSTRUCTIONS:
                - If the user is asking about out to a trip and accomodation concepts, just tell them "من راهنمای سفر هستم و فقط در این زمینه میتونم به شما کمک کنم"
             """
             },
            {"role": "user", "content": user_query}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # Check if the model wants to call a function
        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            if tool_call.function.name == "query_similar_rooms":
                # Parse the arguments
                function_args = json.loads(tool_call.function.arguments)
                # Call the function
                search_results = query_similar_rooms(function_args["query"])
                
                # Add the function response to the messages
                messages.append(response_message)
                tool_response = {
                    "role": "tool",
                    "name": "query_similar_rooms",
                    "content": json.dumps(search_results),
                    "tool_call_id": tool_call.id
                }
                messages.append(tool_response)

                # Get a new response from the model
                # second_response = self.client.chat.completions.create(
                #     model="gpt-4o-mini",
                #     messages=messages
                # )
                
                # Return both the tool response and the model's response
                return {
                    "tool_response": search_results,
                    # "model_response": second_response.choices[0].message.content
                }

        return response_message.content

API_KEY = os.getenv('OPENAI_API_KEY', '')
agent = AccommodationAgent(API_KEY) 

def get_query_from_agent(user_query: str):
    response = agent.process_user_query(user_query)
    return response