from langchain.memory import ConversationSummaryMemory
from langchain_core.language_models import LLM


# Wrapper for your existing Groq function
class CustomLLM(LLM):

    def _call(self, prompt, stop=None):
        from multimodal import get_response
        return get_response(prompt)

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "custom_groq"


# Initialize memory
llm = CustomLLM()

memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True
)


def update_memory(user_input, ai_output):
    memory.save_context(
        {"input": user_input},
        {"output": ai_output}
    )


def get_memory():
    return memory.buffer

#ConversationBufferMemory=Stores all chats