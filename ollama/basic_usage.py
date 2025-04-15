from dotenv import dotenv_values
from langchain_ollama import OllamaLLM

env = dotenv_values("../.env")

# https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html
llm = OllamaLLM(
    model="mistral:latest",
    # optional
    base_url=f"{env["OLLAMA_IP"]}:{env["OLLAMA_PORT"]}", # when Ollama is served not on localhost
    cache=False,
    temperature=1, # default: 0.8, top_k and top_p also available
    verbose=False
)

# no streaming
# llm.invoke("Tell me a joke")

# end with '|' to see the actual tokens
for token in llm.stream("Tell me a joke"):
    print(token, end="|", flush=True)