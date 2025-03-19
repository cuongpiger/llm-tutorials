from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage, AIMessageChunk

from typing import List
import re

class DeepSeek:
    def __init__(self, endpoint: str, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", max_tokens: int = 32700):
        self.endpoint = endpoint
        self.model_name = model_name
        self._llm_model = ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=endpoint,
            model_name=self.model_name,
            max_tokens=max_tokens,
            streaming=True,
            model_kwargs={"stop": ["<|eot_id|>",'<|eom_id|>']},
        )

    @property
    def llm_model(self):
        return self._llm_model

    def rag_chain(self, prompt: ChatPromptTemplate, retriever):
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm_model
            | self._parse
        )

    def _parse(self, ai_message: AIMessage) -> str:
        """Parse the AI message."""
        return re.sub(r".*</think>\s*", "", ai_message.content, flags=re.DOTALL)

