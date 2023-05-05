from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def load_documentation(doc_dir: str):
    """Load Markdown documentation from a directory"""
    path = Path(doc_dir)
    if not path.is_dir:
        raise ValueError("Input path must be a directory")

    markdown_files = path.glob("**/*.md")

    loaders = [
        UnstructuredMarkdownLoader(str(filepath.resolve()))
        for filepath in markdown_files
    ]
    return loaders


question_prompt_template = """You are a help bot for an open source software community.
Use this section of the documentation to answer a user's question, if the text is related
to the question:
{context}
Question: {question}
Please include any Python code that is relevant to the answer.
Relevant text, if any:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following summaries from the Aim documentation, 
collect them into one combined answer for the user's question. If you don't have a good 
answer, just say that you don't know.

Summaries:
{summaries}

Question: {question}

Combined answer, if any:
"""

COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)


if __name__ == "__main__":
    loaders = load_documentation("../aim/docs/source")
    index = VectorstoreIndexCreator().from_loaders(loaders)

    llm = OpenAI(temperature=0.2)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=index.vectorstore.as_retriever(),
        chain_type_kwargs={
            "question_prompt": QUESTION_PROMPT,
            "combine_prompt": COMBINE_PROMPT,
        },
    )

    query = "How doing I use Aim with HuggingFace spaces?"
    print(qa.run(query))
