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


def get_sql_tools():
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()


# --- Input parser ---
def parse_schedule_input(text_input: str):
    name = re.search(r'name[:\s]*([\w\s]+)', text_input, re.IGNORECASE)
    purpose = re.search(r'purpose[:\s]*([\w\s]+)', text_input, re.IGNORECASE)
    country = re.search(r'country[:\s]*([\w\s]+)', text_input, re.IGNORECASE)

    # Support multiple nextimrdate conditions (<=, >=, <, >, =)
    conditions = re.findall(r'nextimrdate\s*(<=|>=|<|>|=)?\s*([\d]{4}-[\d]{2}-[\d]{2})', text_input, re.IGNORECASE)
    date_conditions = [(op if op else "=", dt) for op, dt in conditions]

    if name or purpose or country or date_conditions:
        return {
            "name": name.group(1).strip() if name else None,
            "purpose": purpose.group(1).strip() if purpose else None,
            "country": country.group(1).strip() if country else None,
            "date_conditions": date_conditions
        }

    # Fallback: comma-separated
    parts = [p.strip() for p in text_input.split(",")]
    return {
        "name": parts[0] if len(parts) > 0 else None,
        "purpose": parts[1] if len(parts) > 1 else None,
        "country": parts[2] if len(parts) > 2 else None,
        "date_conditions": [("=", parts[3])] if len(parts) > 3 else []
    }


# --- Schedule insert ---
def create_schedule(user_text: str):
    data = parse_schedule_input(user_text)
    date_conditions = data.get("date_conditions", [])
    created_ids = []
    skipped = []

    with db._engine.connect() as conn:
        if date_conditions:
            # Build WHERE clause dynamically
            where_clauses = []
            params = {}
            for idx, (op, dt) in enumerate(date_conditions):
                param_name = f"date{idx}"
                where_clauses.append(f"nextimrdate {op} :{param_name}")
                params[param_name] = dt

            query = f"SELECT mucid, purpose, country, nextimrdate FROM modelusecase WHERE {' AND '.join(where_clauses)}"
            matches = conn.execute(text(query), params).fetchall()

            for match in matches:
                # Duplicate check
                existing = conn.execute(text("""
                    SELECT scheduleid FROM schedules
                    WHERE name=:name AND purpose=:purpose AND country=:country
                """), {
                    "name": match.mucid,
                    "purpose": match.purpose,
                    "country": match.country
                }).fetchone()
                if existing:
                    skipped.append(existing[0])
                    continue

                # Insert
                schedule_data = {"name": match.mucid, "purpose": match.purpose, "country": match.country}
                result = conn.execute(text("""
                    INSERT INTO schedules (name, purpose, country)
                    VALUES (:name, :purpose, :country)
                    RETURNING scheduleid
                """), schedule_data)
                created_ids.append(result.scalar_one())

            conn.commit()
            if matches:
                return {
                    "status": "ok",
                    "created": created_ids,
                    "skipped_duplicates": skipped,
                    "message": f"✅ {len(created_ids)} schedule(s) created, {len(skipped)} skipped as duplicates."
                }

        # Fallback manual insert
        result = conn.execute(text("""
            INSERT INTO schedules (name, purpose, country)
            VALUES (:name, :purpose, :country)
            RETURNING scheduleid
        """), {"name": data.get("name"), "purpose": data.get("purpose"), "country": data.get("country")})
        created_ids.append(result.scalar_one())
        conn.commit()

    return {"status": "ok", "created": created_ids, "message": f"🆕 Schedule created with ID: {created_ids}"}


def get_insert_tool():
    return Tool(
        name="RAG_Insert",
        func=create_schedule,
        description=(
            "Creates schedule(s). If `nextimrdate` matches records in `modelusecase`, "
            "it will create schedules for all of them, avoiding duplicates. "
            "Supports operators: <=, >=, <, >, =. "
            "If no match, inserts using provided name, purpose, and country."
        )
    )


# --- Usage summary ---
def summarize_usage(query: str):
    with db._engine.connect() as conn:
        result = conn.execute(text("""
            SELECT user_id, user_name, usage_date, total_sessions, total_requests, total_tokens, top_action, last_used_at
            FROM usage_summary;
        """))
        rows = [dict(r._mapping) for r in result]

    if not rows:
        return "No usage summary data found."

    summary_prompt = f"""
        You are a data analyst. Here is the usage_summary data:
        {rows}

        User request: {query}

        🚨 STRICT INSTRUCTIONS:
        - Always output in **Markdown format**.
        - Use bullet points and headings.
        - Numbers must be **bold**.

        # 📊 Usage Summary

        ### Overall Totals
        - **Total Sessions:** X
        - **Total Requests:** X
        - **Total Tokens:** X

        ### 🔹 Averages
        - **Average Sessions per Day:** X
        - **Average Requests per Session:** X
        - **Average Tokens per Request:** X

        ### 🔥 Activity Trends
        - (Write 1–2 bullets)

        ### ⏰ Time-Based Insights
        - (Last active date, peak activity date)

        ### 🏆 Top Users
        1. User A
        2. User B
        3. ...

        ### ❗ Key Takeaways
        - (3 concise bullets)
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
            "- 'Summarize usage_summary for all users where usage_date >= 2025-08-23'\n"
            "- 'Summarize usage for the last 7 days'\n"
            "- 'Summarize usage for user Satish after 2025-08-20'\n\n"
            "The tool will query the database, fetch filtered rows, and produce "
            "a structured report with totals, averages, activity trends, and key takeaways."
        )
    )


# --- Agent assembly ---
tools = [get_rag_tool()] + get_sql_tools() + [get_insert_tool(), get_usage_tool()]

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

    if 'usage_summary' in message:
        final_res = summarize_usage(message)
    else:
        reply = run_agent_query(message)
        final_res = reply["output"]

    return jsonify(status="ok", body={"reply": final_res, "format": "markdown"}), 200


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5003, debug=True)
