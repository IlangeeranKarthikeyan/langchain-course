from dotenv import load_dotenv

from typing import Dict

from langchain_course import IRunnable
from langchain_course import main as run_application
from langchain_course.Summarizer import Summarizer
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


def main(runnable: IRunnable, context: Dict[str, object]) -> None:
    run_application(runnable, context)


if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(model="gpt-5", temperature=0)
    # llm = ChatOllama(model="gemma3:270m", temperature=0)
    # llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0)
    context: Dict[str, object] = {
        "PromptTemplate": PromptTemplate,
        "llm": llm
    }
    main(Summarizer(), context)
