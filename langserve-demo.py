from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langserve import add_routes
import uvicorn

load_dotenv()

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
# print(llm.invoke("Hey , who are you ??"))

system_template = """
    Translate the following into {language}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system",system_template),
     ("user","{text}")]
)

parser = StrOutputParser()
chain = prompt_template | llm | parser

# chain2 = prompt_template | llm

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server"
)

add_routes(
    app,
    chain,
    path="/chain"
)


if __name__=="__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)