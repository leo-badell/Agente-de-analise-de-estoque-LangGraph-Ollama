# Agente de análise de estoque (LangGraph + Ollama)

Agente de IA que interpreta perguntas sobre estoque, consulta a [API de Estoque](https://github.com/wangsouza/api-estoque) por meio de uma ferramenta Python e classifica cada produto como **disponível**, **baixo** ou **esgotado**. O fluxo é orquestrado com LangGraph e usa um LLM local e gratuito via Ollama (`qwen2.5:7b`).

O código do agente está em [notebooks/agente_estoque.ipynb](notebooks/agente_estoque.ipynb).
