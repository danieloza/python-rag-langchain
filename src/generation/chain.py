from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from src.config import settings

def build_llm():
    gen_pipeline = pipeline(
        task="text2text-generation",
        model=settings.LLM_MODEL,
        max_new_tokens=256, # Senior IT: Increased for better chat
        do_sample=True,      # Senior IT: Enable sampling for more natural chat
        temperature=0.7
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain(retriever):
    llm = build_llm()

    # Senior IT: Versatile Hybrid Prompt
    template = """Jesteś inteligentnym i wszechstronnym Asystentem Systemu Danex. 
    
    1. Jeśli pytanie dotyczy faktur lub danych w kontekście - użyj poniższych informacji.
    2. Jeśli pytanie jest ogólne lub dotyczy innych tematów - odpowiedz najlepiej jak potrafisz jako pomocny asystent.
    3. Bądź uprzejmy i profesjonalny.

    Kontekst z bazy wiedzy: {context}

    Pytanie użytkownika: {question}

    Twoja odpowiedź:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
