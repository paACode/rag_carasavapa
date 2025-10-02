# NEED API KEY and Premium OpenAI Account with Credits !!!


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# 1. Load document (PDF)
# Source : https://www.ijcar.net/assets/pdf/Vol6-No5-May2019/07.-Basics-of-Research-Design-A-Guide-to-selecting-appropriate-research-design.pdf
loader = PyPDFLoader(file_path="../data/07.-Basics-of-Research-Design-A-Guide-to-selecting-appropriate-research-design.pdf")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Create embeddings + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Build retrieval-based QA
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=retriever,
    return_source_documents=True
)

# 5. Ask a question
query = "What are convergent parallel mixed methods?"
result = qa.run(query)

print(result)