"""
HomeMatch
---------
End-to-end flow to:
- Load or generate real estate listings
- Index them into two vectorstores (raw vs. LLM-enhanced)
- Collect user preferences and run semantic retrieval
- Optionally augment top listings with personalized copy

Components:
- Raw vectorstore embeds the structured listing JSON as text
- Semantic vectorstore embeds an LLM-enhanced descriptive text

Environment variables expected:
- `OPENAI_API_KEY`: API key for the OpenAI-compatible endpoint
- `OPENAI_API_BASE`: Base URL for the OpenAI-compatible endpoint
"""

import os

os.environ["OPENAI_API_KEY"] = "voc-16184921971266774216506689a68e970a3c0.17240526"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000, api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"])

def get_response(prompt, temperature=0.7):
    """Call the chat LLM with a prompt.

    Args:
        prompt (str): Natural language instruction/content sent to the LLM.
        temperature (float, optional): Decoding temperature to control
            randomness. Defaults to 0.7.

    Returns:
        langchain.schema.messages.AIMessage: LLM response message. The textual
        content is available via `response.content`.
    """
    return llm.invoke(prompt, temperature=temperature)

def main():
    """Run the HomeMatch pipeline.

    Steps:
    1. Read pre-generated listings from `listings.json`
    2. Build two vectorstores: raw text JSON and LLM-enhanced descriptions
    3. Collect user preferences and convert to a query
    4. Retrieve top-K candidates from both stores
    5. Optionally LLM-augment the retrieved listings for personalization
    """

    # Generating Real Estate Listings
    with open("listings.json", "r") as f:
        listings = json.load(f)

    # Storing Listings in a Vector Database
    # Raw vectorstore: embed the full listing JSON (except ID) as `page_content`.
    raw_documents = [
        Document(page_content=json.dumps({k: v for k, v in listing.items() if k != "id"}, ensure_ascii=False), 
        metadata={"id": listing["id"], "neighborhood": listing["neighborhood"], "price": listing["price"]}
        )
        for listing in listings
    ]
    print("Number of raw documents: ", len(raw_documents))
    raw_vectorstore = Chroma.from_documents(
        raw_documents,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="raw_listings"
    )

    with open("semantic_enhanced_listings.txt", "r") as f:
        semantic_enhanced_documents = [Document(page_content=line, metadata={"id": line.split(",")[0].split(":")[1]}) for line in f.readlines()]

    print("Number of semantic enhanced documents: ", len(semantic_enhanced_documents))
    semantic_enhanced_vectorstore = Chroma.from_documents(
        semantic_enhanced_documents,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="semantic_listings"
    )
        
        

    # Building the User Preference Interface
    # For demo purposes, we hardcode questions/answers. In production, collect from a UI.
    questions = [   
                "How big do you want your house to be?",
                "What are 3 most important things for you in choosing this property?", 
                "Which amenities would you like?", 
                "Which transportation options are important to you?",
                "How urban do you want your neighborhood to be?",   
            ]
    answers = [
        "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
        "A quiet neighborhood, good local schools, and convenient shopping options.",
        "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
        "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
    ]

    def preferences_to_query(questions: list, answers: list) -> str:
        """Flatten Q&A pairs into a single natural-language query string."""
        return " \n".join([f"{question} {answer}" for question, answer in zip(questions, answers)])
    
    preferences = preferences_to_query(questions, answers)
    print("Preferences: ", preferences, "\n\n")

    # Searching Based on Preferences
    # Retrieve from raw index (structured JSON text)
    print("Step 5: Searching Based on Preferences")
    similarity_search_raw_vectorstore = raw_vectorstore.similarity_search(preferences, k=2)
    for document in similarity_search_raw_vectorstore:
        print(document.page_content, "\n")

    #Listing Retrieval Logic: Fine-tune the retrieval algorithm to ensure that the most relevant listings are selected based on the semantic closeness to the buyer’s preferences.
    # Retrieve from semantic index (LLM-generated prose). In a more advanced system,
    # you could fetch a larger candidate set and re-rank by domain-specific criteria.
    print("\n\nListing Retrieval Logic")
    similarity_search_semantic_enhanced_vectorstore = semantic_enhanced_vectorstore.similarity_search(preferences, k=2)
    for document in similarity_search_semantic_enhanced_vectorstore:
        print(document.page_content, "\n")

    # Step 6: Personalizing Listing Descriptions
    #LLM Augmentation: For each retrieved listing, use the LLM to augment the description, tailoring it to resonate with the buyer’s specific preferences.
    print("\n\nLLM Augmentation")
    prompt = """
        You are a real estate agent. You are given a listing and a user preferences. You need to change craete a listing so its more appealing to the user.
        Without changing the factual information of the listing, enhance or enfasize the aspects and use words that resonate with the user preferences.

        The listing propertie description is:
        {listing_description}

        The user preferences are:
        {user_preferences_description}

        Return the final description in a single string.
        """
    for document in similarity_search_raw_vectorstore:
        # Personalize the listing description using the user's preferences
        response = get_response(prompt.format(listing_description=document.page_content, user_preferences_description=preferences), temperature=0.5)
        print("Listing Enhanced: ", document.metadata["neighborhood"], document.metadata["price"])
        print(response.content, "\n")

    # Step 7: Deliverables and Testing

    # Step 8: Project Submission

if __name__ == "__main__":
    main()