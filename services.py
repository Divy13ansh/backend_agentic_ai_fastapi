import json
import re
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
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
    template="""You are a precise assistant for generating charts/diagrams.  
    Rules:  
    - For mermaid: return pure mermaid syntax (no markdown, no code blocks).  
    - For charts: return only Chart.js JSON config.  
    - For other libraries: return only their raw code syntax.  
    - Never add explanations, comments, or text. Return code only.  

    Question: {question}  
    Context: {context}  
    Answer:""",
    input_variables=["question", "context"],
)


decision_prompt = PromptTemplate(
    template="""You are an AI CFO assistant responsible for **strategic business decisions**.  
    You will be given a query and supporting company data.  

    Your task:  
    - Provide a clear **decision or recommendation** directly addressing the query.  
    - Justify it using evidence from the descriptive answer and chart context (figures, patterns, trends).  
    - Be concise but specific about what the company should do.  
    - If the data is insufficient, say so and suggest what extra info is needed.  

    Query: {question}  
    Company Data & Context: {context}  
    Decision:""",
    input_variables=["question", "context"],
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
    formatting_prompt = f"""
    You are a response formatter. Rewrite the following answer into a **concise, user-friendly format**.

    Query: {query}
    Answer: {answer}

    Rules:
    - Use bullet points or very short paragraphs
    - Keep it conversational and easy to read
    - Highlight only the most important insights
    - Remove unnecessary details or repetition
    """
    
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

def process_query(query, dataset_name="Zomato"):
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
                    # "title": doc.metadata.get("title", "Unknown"),
                    # "source": doc.metadata.get("source", "Unknown"),
                    "pages": doc.metadata.get("page", [])
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
    try:
        # Stage 1: Descriptive answer from business dataset
        data_store = get_vector_store(dataset_name)
        results = data_store.similarity_search_with_score(query, k=3)

        if not results or max(score for _, score in results) < 0.37:
            return {
                "response": [{"message": f"No information found in dataset '{dataset_name}'"}]
            }

        data_qa = get_qa_chain(dataset_name, data_prompt)
        stage1 = data_qa({"query": query})
        descriptive_answer = stage1["result"]

        # Extra context: pull from context_collection (diagram/chart RAG DB)
        context_store = get_vector_store("context_collection")
        context_results = context_store.similarity_search(query, k=3)
        chart_context = "\n\n".join([doc.page_content for doc in context_results])

        # Stage 3: Decision making, with both descriptive + chart context
        decision_input = decision_prompt.format(
            question=query,
            context=f"{descriptive_answer}\n\nAdditional Context (Charts/Diagrams):\n{chart_context}\nADD A FEW RELEVANT EMOJIS"
        )
        stage3 = llm.invoke(decision_input)

        # Metadata
        metadata = {
            "source_documents": [
                {
                    # "title": doc.metadata.get("title", "Unknown"),
                    # "source": doc.metadata.get("source", "Unknown"),
                    "pages": doc.metadata.get("page", [])
                }
                for doc in stage1["source_documents"]
            ],
            "dataset_used": dataset_name
        }

        # Final response
        formatted = {
            "query": query,
            "descriptive_answer": descriptive_answer,
            "decision": stage3.content.strip(),
            "metadata": metadata,
        }
        return {"response": formatted}

    except Exception as e:
        return {"error": str(e)}
