import os

os.environ["OPENAI_API_KEY"] = "voc-16184921971266774216506689a68e970a3c0.17240526"
os.environ["OPENAI_BASE"] = "https://openai.vocareum.com/v1"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
from langchain_core.documents import Document

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=1000, api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE"])

def get_response(prompt, temperature=0.5):
    return llm.invoke(prompt, temperature=temperature)

def generate_enhanced_listings(listings):
    semantic_enhanced_documents = []
    # Prompt to turn numeric and terse details into richer human-friendly prose
    prompt = """
        You are a real estate agent. Convert the numeric description of the listings into descriptive text. Ignore the id field.
        For example:
        "3 bedrooms, 2 bathrooms, 1,500 sqft" -> "A comfortable three-bedroom house with a spacious kitchen and a cozy living room."
        "2 bedrooms, 1 bathroom, 1,000 sqft" -> "A cozy two-bedroom house with a small kitchen and a cozy living room."

        The listing properties are:
        {listing_properties}

        Return the final description in a single string.
        """
    for listing in listings:
        # Ask the LLM to produce a descriptive sentence for the current listing
        response = get_response(prompt.format(listing_properties=json.dumps(listing)), temperature=0.5)
        if "```json" in response.content:
            response.content = response.content.replace("```json", "").replace("```", "")
        # Build an enhanced document focused on readable description text
        semantic_enhanced_documents.append(
            Document("id: "+str(listing["id"])+ ", converted description: " + response.content + ", original description: "+listing["description"] + ", neighborhood_description: "+listing["neighborhood_description"], metadata={"id": listing["id"]})
        )
    return semantic_enhanced_documents

if __name__ == "__main__":
    with open("listings.json", "r") as f:
        listings = json.load(f)
    semantic_enhanced_documents = generate_enhanced_listings(listings)
    with open("semantic_enhanced_listings.txt", "w") as f:
        for document in semantic_enhanced_documents:
            f.write(document.page_content + "\n")