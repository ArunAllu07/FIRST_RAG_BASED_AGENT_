from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

import chromadb
from sentence_transformers import SentenceTransformer
from ddgs import DDGS


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. INITIALIZE GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ============================================================
# 3. CALCULATOR TOOL
# ============================================================

@tool
def calculator(a: float, b: float) -> float:
    """Multiply two numbers."""

    return a * b


# ============================================================
# 4. WEB SEARCH TOOL
# ============================================================

@tool
def web_search(query: str) -> str:
    """Search the web for current or recent information."""

    try:
        results = DDGS().text(
            query,
            max_results=3
        )

        if not results:
            return "No search results found."

        output = []

        for result in results:
            output.append(
                f"Title: {result.get('title', '')}\n"
                f"Information: {result.get('body', '')}\n"
                f"URL: {result.get('href', '')}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Web search failed: {str(e)}"


# ============================================================
# 5. LOAD CHROMADB
# ============================================================

print("Loading PDF vector database...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="surface_notes"
)

print(
    f"PDF database loaded successfully. "
    f"Documents: {collection.count()}"
)


# ============================================================
# 6. PDF SEARCH / RAG TOOL
# ============================================================

@tool
def search_pdf(query: str) -> str:
    """
    Search the SURFACE engineering PDF for relevant information.
    Use this tool when the user asks questions related to the PDF.
    """

    try:
        # Convert question into embedding
        query_embedding = embedding_model.encode(
            [query]
        ).tolist()

        # Search ChromaDB
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )

        documents = results.get("documents", [[]])[0]

        if not documents:
            return "No relevant information found in the PDF."

        # Combine retrieved chunks
        return "\n\n".join(documents)

    except Exception as e:
        return f"PDF search failed: {str(e)}"


# ============================================================
# 7. SYSTEM PROMPT
# ============================================================

system_prompt = """
You are a helpful AI assistant.

You have access to three tools:

1. calculator
   - Use this tool when the user needs multiplication.
   - Example: 25 × 40.

2. web_search
   - Use this tool when the user asks for current, recent,
     or web-based information.

3. search_pdf
   - Use this tool when the user asks about information
     contained in the SURFACE engineering PDF.
   - The PDF contains surface engineering notes.
   - Use the retrieved PDF information to answer the question.
   - Do not invent information that is not supported by
     the retrieved PDF content.

Use the appropriate tool whenever necessary.

You also have conversation memory.
Use previous messages to understand follow-up questions
within the same conversation.

Give clear and simple answers.
"""


# ============================================================
# 8. CREATE AGENT + SQLITE MEMORY
# ============================================================

with SqliteSaver.from_conn_string("agent_memory.db") as memory:

    agent = create_agent(
        model=llm,
        tools=[
            calculator,
            web_search,
            search_pdf
        ],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    print("\n========================================")
    print("        MINI AI AGENT")
    print("========================================")
    print("Tools available:")
    print("1. Calculator")
    print("2. Web Search")
    print("3. SURFACE PDF Search")
    print("4. Conversation Memory")
    print("----------------------------------------")
    print("Type 'exit' to stop.")
    print("========================================\n")


    # ========================================================
    # 9. INTERACTIVE CHAT
    # ========================================================

    while True:

        user_input = input("You: ")

        if user_input.lower().strip() == "exit":
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        try:

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": "arun_chat"
                    }
                }
            )

            # Get final assistant response
            response = result["messages"][-1].content

            print("\nAgent:", response)
            print()

        except Exception as e:

            print("\nError:", str(e))
            print()