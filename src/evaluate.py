import json
from src.retrieval.engine import get_retriever
from src.generation.chain import get_qa_chain

def evaluate_rag():
    print("--- Starting Senior RAG Evaluation (LCEL) ---")
    retriever = get_retriever()
    qa_chain = get_qa_chain(retriever)
    
    test_questions = [
        "What programming languages does Daniel use?",
        "Where does Daniel study?"
    ]
    
    results = []
    for q in test_questions:
        # In LCEL, we directly get the string answer
        answer = qa_chain.invoke(q)
        results.append({
            "question": q,
            "answer": answer
        })
        
    print("\n[SUCCESS] Evaluation results generated:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    evaluate_rag()
