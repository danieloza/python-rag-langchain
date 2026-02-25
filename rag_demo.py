import sys
import io
# Senior IT Fix: Force UTF-8 for console output to handle emojis correctly on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.retrieval.engine import get_retriever
from src.generation.chain import get_qa_chain
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", help="Ask a question")
    args = parser.parse_args()

    retriever = get_retriever()
    qa_chain = get_qa_chain(retriever)

    q = args.question or "What programming languages does Daniel use?"
    
    # Senior IT Fix: Invoke with string directly
    res = qa_chain.invoke(q)

    # Print with clear markers for the bridge to parse
    print(f"---RAG-START---")
    print(f"Q: {q}")
    print(f"A: {res}")
    print(f"---RAG-END---")

if __name__ == '__main__':
    main()
