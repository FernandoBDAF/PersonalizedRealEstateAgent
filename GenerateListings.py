"""
GenerateListings
----------------
Utility script to synthesize mock real estate listings using an LLM and save
them to a JSON file (`listings.json`). This is intended for demo/testing
purposes to seed downstream vector indexing and semantic search flows.

Environment variables expected:
- `OPENAI_API_KEY`: API key for the OpenAI-compatible endpoint
- `OPENAI_BASE`: Base URL for the OpenAI-compatible endpoint

Note: Credentials are read from environment variables (and optionally a `.env`
file if present). Do not commit real secrets to source control.
"""

import os

os.environ["OPENAI_API_KEY"] = "voc-16184921971266774216506689a68e970a3c0.17240526"
os.environ["OPENAI_BASE"] = "https://openai.vocareum.com/v1"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=1000, api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE"])

def get_response(prompt, temperature=0.5):
    """Call the chat LLM with a prompt.

    Args:
        prompt (str): Natural language instruction/content sent to the LLM.
        temperature (float, optional): Decoding temperature to control
            randomness. Defaults to 0.5.

    Returns:
        langchain.schema.messages.AIMessage: LLM response message. The textual
        content is available via `response.content`.
    """
    return llm.invoke(prompt, temperature=temperature)

def generate_listings(amount=5):
    """Generate a list of mock real estate listings and write to JSON.

    The LLM is prompted to return a single JSON object per listing. We parse the
    response and aggregate into a Python list, then persist to `listings.json`.

    Args:
        amount (int, optional): Number of listings to generate. Defaults to 5.

    Side effects:
        Writes the resulting list to `listings.json` in the current directory.
    """
    listings = []
    trys = 0
    while len(listings) < amount and trys < amount + 10:
        trys += 1
        prompt = f"""Generate a real estate listing with mock data and responde in with a single JSON object with the following fields: 
        neighborhood, price, bedrooms, bathrooms, size, description and neighborhood description. For exemple:
        {{
            "neighborhood": "Green Oaks",
            "price": "$800,000",
            "bedrooms": "3",
            "bathrooms": "2",
            "size": "2,000 sqft",
            "description": "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
            "neighborhood_description": "Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze."
        }}
        """
        # Query the LLM for a single listing in JSON format
        response = get_response(prompt, temperature=0.85)
        if "```json" in response.content:
            response.content = response.content.replace("```json", "").replace("```", "")
        else:
            print(f"Error parsing response {trys}: {response.content}")
            continue

        try:
            # Parse the JSON object returned by the LLM
            parsed_response = json.loads(response.content)
        except json.JSONDecodeError:
            print(f"Error parsing response {trys}: {response.content}")
            continue

        # Normalize the listing into a consistent dict structure
        listings.append({
            "id": trys,
            "neighborhood": parsed_response["neighborhood"],
            "price": parsed_response["price"],
            "bedrooms": parsed_response["bedrooms"],
            "bathrooms": parsed_response["bathrooms"],
            "size": parsed_response["size"],
            "description": parsed_response["description"],
            "neighborhood_description": parsed_response["neighborhood_description"]
        })

    print("Number of listings: ", len(listings))
    
    # Persist generated listings for downstream steps
    with open("listings.json", "w") as f:
        json.dump(listings, f)

if __name__ == "__main__":
    generate_listings(100)