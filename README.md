# Retrieval_Augmented_Generation-Human_body-Q-and-A
# 🧠 Human Body Q&A using RAG (Retrieval-Augmented Generation)

This project is a **beginner-friendly RAG (Retrieval-Augmented Generation) pipeline** built in Python using **Hugging Face, FAISS, and LangChain**.

It allows you to **ask questions about the human body** and get answers generated from a **PDF document (`human_body.pdf`)**, instead of from the model’s general knowledge.

---

## 📌 Features

- 🔍 Loads and reads **PDF document** (`human_body.pdf`)
- ✂️ Splits the text into small **chunks** for better retrieval
- 🧮 Uses **Sentence-Transformer embeddings** to convert text into vectors
- 📦 Stores vectors in a **FAISS vector store**
- 🤖 Uses **FLAN-T5 (Hugging Face)** as the LLM for answer generation
- ❓ Answers **user questions** based only on the PDF content (RAG)
- 🎓 Designed to be **simple, clean, and easy to understand** for beginners

---

## 🛠 Tech Stack

- **Language:** Python
- **Libraries:**
  - `langchain-community`
  - `langchain-text-splitters`
  - `sentence-transformers`
  - `faiss-cpu`
  - `pypdf`
  - `transformers`
  - `torch`
- **Model:**
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - LLM: `google/flan-t5-base` (via Hugging Face `pipeline`)

---

## 📂 Project Structure

Example structure:

```bash
.
├── human_body.pdf        # Source document for Q&A
├── rag_human_body.ipynb  # Main notebook (Colab / Jupyter)
└── README.md             # Project description
You can rename the files as you like, but make sure the PDF file name in the code matches the actual file.

✅ Setup Instructions
1️⃣ Clone the Repository
bash
Copy code
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
2️⃣ Install Dependencies
If you are running locally:

bash
Copy code
pip install langchain-community langchain-text-splitters sentence-transformers faiss-cpu pypdf transformers accelerate torch
If you are using Google Colab, use:

python
Copy code
!pip install -q langchain-community langchain-text-splitters sentence-transformers faiss-cpu pypdf transformers accelerate
⚠️ Some warnings about requests version may appear in Colab.
They are safe to ignore for this project.

📘 RAG Pipeline Code (Core Logic)
Below is the core RAG pipeline used in this project.

1️⃣ Imports
python
Copy code
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
2️⃣ Load PDF
python
Copy code
loader = PyPDFLoader("human_body.pdf")   # Ensure this file is in the same folder
docs = loader.load()
3️⃣ Split into Chunks
python
Copy code
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)
4️⃣ Create Embeddings + Vector Store
python
Copy code
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
5️⃣ Load LLM (Hugging Face Pipeline)
python
Copy code
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

hf = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)
6️⃣ Simple RAG Function
python
Copy code
def rag_answer(question):
    # 1) Retrieve the most relevant chunk from vector store
    result = vectorstore.similarity_search(question, k=1)[0]
    context = result.page_content

    # 2) Build prompt for the model
    prompt = (
        "Answer the question using only the text below.\n\n"
        "Text:\n" + context + "\n\n"
        "Question: " + question + "\n"
        "Answer in short and simple English."
    )

    # 3) Generate answer from LLM
    output = hf(prompt, max_new_tokens=150)
    return output[0]["generated_text"]
7️⃣ Ask Questions
python
Copy code
question = "What are the main systems in the human body?"
answer = rag_answer(question)

print("Q:", question)
print("A:", answer)
You can ask more questions like:

python
Copy code
print(rag_answer("What does the brain do?"))
print(rag_answer("Explain the circulatory system."))
💡 What I Learned
How to build a basic RAG pipeline using Python

How to:

Load and process PDF files

Create text chunks with RecursiveCharacterTextSplitter

Generate embeddings using Sentence Transformers

Store and search vectors using FAISS

Use a Hugging Face text generation pipeline (FLAN-T5)

Combine retrieval + generation to answer questions from a document

🚀 Future Improvements
Add a Streamlit UI for interactive Q&A

Support multiple PDFs

Store the vector store on disk and load it later

Add better prompts and answer formatting

Add support for GPU for faster inference (if available)

🧑‍💻 Author
Name: [Mangaiyarkarasi]

Role: Aspiring Data Scientist / AI Engineer

GitHub: https://github.com/<your-username>

LinkedIn: [https://www.linkedin.com/feed/update/urn:li:activity:7403043371721932800/]

