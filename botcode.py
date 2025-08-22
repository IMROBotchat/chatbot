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

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 1. Load env and set up heavy singletons once
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
DB_URI = os.getenv("DATABASE_URI")

# 2. Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

# 3. Build or load vectorstore once
def build_retriever():
    docs = TextLoader("imro_docs.txt").load()
    chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    embed_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = "chroma_db"
    if os.path.isdir(persist_dir):
        store = Chroma(persist_directory=persist_dir, embedding_function=embed_fn)
    else:
        store = Chroma.from_documents(chunks, embedding=embed_fn, persist_directory=persist_dir)
        store.persist()
    return store.as_retriever()

retriever = build_retriever()

# 4. Prepare RAG chain & SQL toolkit once
rag_tool = Tool(
    name="RAG_Search",
    func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever).run,
    description="Perform a retrieval-augmented semantic query over the IMRO document and database tables corpus."
)

db = SQLDatabase.from_uri(DB_URI)
sql_tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

# --- Schedule creation helpers ---

def create_schedule(text_input: str) -> str:
    """Insert a single schedule from free-text input."""
    data = parse_schedule_input(text_input)
    insert_sql = text("""
        INSERT INTO schedules (name, purpose, country)
        VALUES (:name, :purpose, :country)
        RETURNING scheduleid;
    """)
    with db._engine.begin() as conn:
        new_id = conn.execute(insert_sql, data).scalar_one()
    return f"✅ Schedule created with ID: {new_id}"

def parse_schedule_input(text_input: str) -> dict:
    # Try explicit field: value first
    def capture(field):
        m = re.search(fr"{field}[:\s]+([^\n,]+)", text_input, re.IGNORECASE)
        return m.group(1).strip() if m else None

    name = capture("name")
    purpose = capture("purpose")
    country = capture("country")

    # If nothing matched, you could add NLP or keyword heuristics here
    return {
        "name": name or "unspecified",
        "purpose": purpose or "unspecified",
        "country": country or "unspecified"
    }

def insert_dispatcher(input_str: str):
    date_match = re.search(r"'?(\d{4}-\d{2}-\d{2})'?", input_str)
    if date_match:
        return "\n".join(create_schedules_for_date(date_match.group(1)))
    else:
        return create_schedule(input_str)  # only for manual inserts

def create_schedules_for_date(target_date: str):
    sql = text("""
        INSERT INTO schedules (name, purpose, country)
        SELECT m.mucid, m.purpose, m.country
        FROM modelusecase m
        WHERE m.nextimrdate = :dt
          AND m.mucid IS NOT NULL AND m.mucid <> ''
          AND m.purpose IS NOT NULL AND m.purpose <> ''
          AND m.country IS NOT NULL AND m.country <> ''
          AND NOT EXISTS (
              SELECT 1 FROM schedules s WHERE s.name = m.mucid
          )
        RETURNING scheduleid, name;
    """)
    with db._engine.begin() as conn:
        rows = conn.execute(sql, {"dt": target_date}).fetchall()
    return [f"✅ Created schedule ID: {sid} for {name}" for sid, name in rows]



insert_tool = Tool(
    name="RAG_Insert",
    func=insert_dispatcher,
    description=(
        "Create a new schedule in the database. "
        "If given a date in YYYY-MM-DD format, bulk-create schedules from modelusecase for that date. "
        "Otherwise, parse name, purpose, and country from the text and insert a single schedule."
    )
)

# 5. Initialize the agent once
tools = [rag_tool] + sql_tools + [insert_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True
)

@app.route("/api/chat", methods=["POST"])
def chat():
    if not request.is_json:
        return jsonify(error="Invalid or missing JSON"), 400
    payload = request.get_json()
    message = payload.get("message", "").strip()
    if not message:
        return jsonify(error="The 'message' field is required"), 400
    result = agent.invoke({"input": message, "chat_history": []})
    return jsonify(status="ok", body={"reply": result["output"]}), 200

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5003, debug=False)
