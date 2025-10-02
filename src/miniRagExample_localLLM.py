from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

instruction_tuned = True

# 1. Load PDF
loader = PyPDFLoader(file_path="../data/07.-Basics-of-Research-Design-A-Guide-to-selecting-appropriate-research-design.pdf")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)


if instruction_tuned :

    # 4. Instruction-tuned local LLM (CPU-friendly)
    pipe = pipeline(
        "text-generation",
        model="distilgpt2",       # small CPU-friendly model
        max_new_tokens=256,       # <-- use max_new_tokens instead of max_length
        do_sample=True,
        temperature=0.7
    )

else:
    # 4. Small local LLM
    pipe = pipeline(
        "text2text-generation",          # <-- changed pipeline type
        model="google/flan-t5-small",    # <-- changed model
        max_new_tokens=256               # <-- keep this
    )

llm = HuggingFacePipeline(pipeline=pipe)

# 5. Build retrieval QA
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 6. Ask a question
query = "What are the three main types of research designs discussed in the article 'Basics of Research Design: A Guide to selecting appropriate research design'?"
result = qa.invoke({"query": query})
print(result["result"])
