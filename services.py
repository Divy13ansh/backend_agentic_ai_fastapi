import json
import re
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate
from config import *
# from dotenv import load_dotenv
# load_dotenv()

# Initialize components once
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT,
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
    temperature=0,
)

# Prompts
data_prompt = PromptTemplate(
    template="""You are an AI financial assistant. 
    Use the retrieved documents to provide a **detailed, structured, and descriptive** answer to the user's query. 
    - Incorporate all relevant details from the context. 
    - Use clear explanations, examples, and numerical insights if available. 
    - If the context is missing information, state it explicitly instead of guessing.

    Question: {question}
    Context: {context}
    Answer:""",
    input_variables=["question", "context"],
)


context_prompt = PromptTemplate(
    template="""Generate charts/diagrams code only.
    - For mermaid: pure mermaid syntax
    - For charts: Chart.js JSON config
    - No explanations, just code
    Question: {question}
    Context: {context}
    Answer:""",
    input_variables=["question", "context"],
)

decision_prompt = PromptTemplate(
    template="""You are an AI CFO assistant responsible for making strategic business decisions. 
    You will be given a query and supporting company data. 

    Your task:
    - Provide a clear **decision or recommendation** directly addressing the query. 
    - Justify the decision using evidence from the data (figures, trends, or patterns). 
    - Be concise but specific, focusing on what the company should do. 
    - If the data is insufficient, state the limitation clearly and suggest what additional data is needed.

    Query: {query}
    Company Data: {context}
    Decision:""",
    input_variables=["query", "context"],
)

nl_prompt = PromptTemplate(
    template="""You are an AI assistant that explains financial and business data to a human user in clear, natural language. 
    You will be given structured information from company data analysis. 

    Guidelines:
    - Convert the analysis into a fluent, easy-to-understand explanation. 
    - Use natural language, avoid technical jargon unless necessary. 
    - Summarize the key points clearly and concisely. 
    - Highlight trends, risks, and opportunities in plain terms. 
    - If numbers are present, keep them but explain their meaning.

    Data Analysis: {context}
    User-Friendly Explanation:""",
        input_variables=["context"],
)

def get_vector_store(dataset_name="my_json_collection"):
    return QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=dataset_name
    )

def get_qa_chain(dataset_name, prompt, k=5):
    vector_store = get_vector_store(dataset_name)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

def clean_chart_code(code):
    if "mermaid" in code.lower():
        code = re.sub(r'```\n?', '', code)
    return code.strip()

def format_response(query, answer, chart_code, metadata):
    # Simple AI formatting
    formatting_prompt = f"""Make this concise (max 6 lines):
    Query: {query}
    Answer: {answer}
    Make it bullet points and conversational."""
    
    try:
        formatted = llm.invoke(formatting_prompt).content.strip()
    except:
        formatted = answer[:300] + "..."
    
    response = [{"message": formatted}]
    
    # Add chart if needed
    if chart_code and ("chart" in query.lower() or "diagram" in query.lower() or "mermaid" in query.lower()):
        if "mermaid" in query.lower():
            response.append({"mermaid": clean_chart_code(chart_code)})
        else:
            response.append({"chart": clean_chart_code(chart_code)})
    
    response.append({"metadata": metadata})
    return response

def process_query(query, dataset_name="my_json_collection"):
    try:
        # Check if we have data
        data_store = get_vector_store(dataset_name)
        results = data_store.similarity_search_with_score(query, k=3)
        
        if not results or max(score for _, score in results) < 0.37:
            return {
                "response": [{"message": f"No information found in dataset '{dataset_name}'"}]
            }
        
        # Get answer
        data_qa = get_qa_chain(dataset_name, data_prompt)
        stage1 = data_qa({"query": query})
        
        # Get chart context
        context_qa = get_qa_chain("context_collection", context_prompt, k=3)
        stage2 = context_qa({"query": stage1["result"]})
        
        # Metadata
        metadata = {
            "source_documents": [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "pages": doc.metadata.get("pages", [])
                }
                for doc in stage1["source_documents"]
            ],
            "dataset_used": dataset_name
        }
        
        # Format response
        formatted = format_response(query, stage1["result"], stage2["result"], metadata)
        return {"response": formatted}
        
    except Exception as e:
        return {"error": str(e)}

def list_datasets():
    try:
        collections = client.get_collections()
        return {"datasets": [col.name for col in collections.collections]}
    except Exception as e:
        return {"error": str(e)}

def decision_maker(query, dataset_name="my_json_collection"):
    # Simple decision maker based on keywords
    try:
        # Check if we have data
        data_store = get_vector_store(dataset_name)
        results = data_store.similarity_search_with_score(query, k=3)
        
        if not results or max(score for _, score in results) < 0.37:
            return {
                "response": [{"message": f"No information found in dataset '{dataset_name}'"}]
            }
        
        # Get answer
        data_qa = get_qa_chain(dataset_name, data_prompt)
        stage1 = data_qa({"query": query})
        
        # Get chart context
        decision_qa = get_qa_chain("context_collection", decision_prompt, k=3)
        stage2 = decision_qa({"query": stage1["result"]})

        # Metadata
        metadata = {
            "source_documents": [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "pages": doc.metadata.get("pages", [])
                }
                for doc in stage1["source_documents"]
            ],
            "dataset_used": dataset_name
        }
        
        # Format response
        formatted = format_response(query, stage1["result"], stage2["result"], metadata)
        return {"response": formatted}
        
    except Exception as e:
        return {"error": str(e)}
    
