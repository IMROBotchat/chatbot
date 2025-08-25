from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import logging
from dotenv import load_dotenv
from sqlalchemy import text
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Load environment variables ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
DB_URI = os.getenv("DATABASE_URI")
os.environ["OPENAI_API_KEY"] = API_KEY

# --- LLM ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- Determine device automatically ---
device = "cuda" if torch.cuda.is_available() else "cpu"


# --- Document retriever (RAG) ---
def get_retriever(doc_paths: list[str], persist_path: str):
    # Check if Chroma DB exists
    if os.path.exists(persist_path):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device})
        vectorstore = Chroma(persist_directory=persist_path, embedding_function=embeddings)
        return vectorstore.as_retriever()

    # Load documents and split
    docs = [doc for path in doc_paths for doc in TextLoader(path).load()]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device})
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_path)
    vectorstore.persist()
    return vectorstore.as_retriever()


retriever = get_retriever(["imro_docs.txt"], "chroma_db")


def get_rag_tool():
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return Tool(
        name="RAG_Search",
        func=qa_chain.run,
        description="Searches internal documents using semantic retrieval"
    )


# --- SQL DB ---
db = SQLDatabase.from_uri(DB_URI)


def get_sql_tool():
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()


# Define schema for Markdown summary
# response_schemas = [
#     ResponseSchema(name="overall_totals", description="Markdown section for overall totals"),
#     ResponseSchema(name="averages", description="Markdown section for averages"),
#     ResponseSchema(name="activity_trends", description="Markdown bullets for activity trends"),
#     ResponseSchema(name="time_based", description="Markdown section for time-based insights"),
#     ResponseSchema(name="top_users", description="Markdown ranked list of top users"),
#     ResponseSchema(name="key_takeaways", description="Markdown list of 3 concise insights"),
#     ResponseSchema(name="recommendations", description="Markdown list of 2 user-level recommendations"),
# ]
#
# parser = StructuredOutputParser.from_response_schemas(response_schemas)
# format_instructions = parser.get_format_instructions()
#
# prompt = PromptTemplate(
#     template="""
#         You are a data analyst. Analyze the following usage_summary data:
#
#         {rows}
#
#         User request:
#         {query}
#
#         Return the analysis strictly in the schema below:
#
#         {format_instructions}
#         """,
#     input_variables=["rows", "query"],
#     partial_variables={"format_instructions": format_instructions},
#     )
#
#
# def summarize_usage(query: str):
#     with db._engine.connect() as conn:
#         result = conn.execute(text("SELECT * FROM usage_summary;"))
#         rows = [dict(r._mapping) for r in result]
#
#     response = llm.invoke(prompt.format(rows=rows, query=query))
#     parsed = parser.parse(response.content)
#
#     # Merge into one Markdown string
#     markdown_report = f"""
#         # 📊 Usage Summary
#
#         #### Overall Totals
#         {parsed['overall_totals']}
#
#         #### 🔹 Averages
#         {parsed['averages']}
#
#         #### 🔥 Activity Trends
#         {parsed['activity_trends']}
#
#         #### ⏰ Time-Based Insights
#         {parsed['time_based']}
#
#         #### 🏆 Top Users
#         {parsed['top_users']}
#
#         #### ❗ Key Takeaways
#         {parsed['key_takeaways']}
#
#         #### 🚨 Recommendations at user level
#         {parsed['recommendations']}
#     """
#     return markdown_report.strip()


# --- Usage summary ---
def summarize_usage(query: str):
    with db._engine.connect() as conn:
        result = conn.execute(text("""
            SELECT * FROM usage_summary;
        """))
        rows = [dict(r._mapping) for r in result]

    if not rows:
        return "No usage summary data found."

    summary_prompt = f"""
        You are a data analyst. You are given the following rows from `usage_summary`:

        {rows}

        The user’s request is:
        {query}

        ---

        ### STRICT INSTRUCTIONS
        - You MUST perform calculations directly from the given rows.
        - You MUST apply any filters/groupings mentioned in the user request.
        - You MUST output in the EXACT Markdown structure below.
        - DO NOT add extra sections.
        - DO NOT output a narrative paragraph.
        - DO NOT leave placeholders like X.
        - If data is missing, output "Not available".
        - Provide usage summary as in a comprehensive, readable and reporting format.

        ---

        ### 📊 Usage Summary

        #### Overall Totals
        - **Total Sessions:** <calculated_value>
        - **Total Requests:** <calculated_value>
        - **Total Tokens:** <calculated_value>

        #### 🔹 Averages
        - **Average Sessions per Day:** <calculated_value>
        - **Average Requests per Session:** <calculated_value>
        - **Average Tokens per Request:** <calculated_value>
        
        #### 🔹 User session Averages
        - **Average sessions in a day for all users :** <calculated_value>
        
        #### 🔥 Activity Trends
        - <1–2 concise bullets about trends in sessions, requests, or tokens>

        #### ⏰ Time-Based Insights
        - Last active date: <latest_date>
        - Peak activity date: <date_with_max_requests_or_tokens>

        #### 🏆 Top Users
        1. <user_name> (<metric>)
        2. <user_name> (<metric>)
        3. <user_name> (<metric>)

        #### ❗ Key Takeaways
        - <3 concise insights>

        #### 🚨 Recommendations at user level
        - <2 actionable recommendations>
    """
    response = llm.invoke(summary_prompt)
    return response.content.strip()


def get_usage_tool():
    return Tool(
        name="Usage_Summary",
        func=summarize_usage,
        description=(
            "Summarizes user activity from the usage_summary table. "
            "You can include filters in your query, such as:\n"
            "- 'Summarize usage_summary for all users group by teams.\n"
            "- 'Summarize usage for the last 7 days'\n"
            "The tool will query the database, fetch filtered rows, and produce "
            "a structured report with totals, averages, activity trends, and key takeaways and reccomondations to improve usage"
            "Provide usage summary as in a comprehensive, readable and reporting format."
        ),
        # return_direct=True
    )


# --- Agent assembly ---
tools = [get_rag_tool()] + get_sql_tool() + [get_usage_tool()]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True
)


def run_agent_query(prompt: str, history=None):
    return agent.invoke({"input": prompt, "chat_history": history or []})


@app.route("/api/chat", methods=["POST"])
def chat():
    if not request.is_json:
        return jsonify(error="Invalid or missing JSON"), 400

    message = request.get_json().get("message", "").strip()
    if not message:
        return jsonify(error="The 'message' field is required"), 400

    reply = run_agent_query(message)
    # final_res = reply["output"]

    return jsonify(status="ok", body={"reply": reply["output"], "format": "markdown"}), 200


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5003, debug=True)
