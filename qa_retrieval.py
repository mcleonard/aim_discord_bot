from pathlib import Path

from langchain.callbacks import AimCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

aim_callback = AimCallbackHandler(
    repo=".",
    experiment_name="Aim Documents Bot",
)


question_prompt_template = """You are a help bot for an open source software community.
Use this section of the documentation to answer a user's question, if the text is related
to the question:
{context}
Question: {question}
Please include any Python code that is relevant to the answer.
Relevant text, if any:"""


combine_prompt_template = """Given the following summaries from the documentation, 
collect them into one combined answer for the user's question. If you don't have a good 
answer, just say that you don't know.
Summaries:
{summaries}
Question: {question}
Combined answer, if any:
"""


def build_qa(doc_dir: str):
    loaders = load_documentation(doc_dir)
    index = VectorstoreIndexCreator().from_loaders(loaders)

    QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )

    llm = OpenAI(temperature=0.2, callbacks=[aim_callback])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=index.vectorstore.as_retriever(),
        chain_type_kwargs={
            "question_prompt": QUESTION_PROMPT,
            "combine_prompt": COMBINE_PROMPT,
        },
        callbacks=[aim_callback]
    )

    return qa


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

if __name__ == "__main__":
    qa = build_qa("../aim/docs/source")

    print(qa.run("What data is captured as text?"))

    aim_callback.flush_tracker(langchain_asset=qa, reset=False, finish=True)