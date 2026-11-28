"""Fixtures compartilhadas.

- Nenhum teste fala com a API real (porta 8000) nem grava em api-estoque/data/produtos.json.
- Fora dos testes marcados com `ollama`, nenhum teste chama o LLM real.
"""

import os
import shutil
from pathlib import Path

import httpx
import pytest

import agente

API_ESTOQUE_DIR = Path(os.getenv("API_ESTOQUE_DIR", Path(__file__).resolve().parents[2] / "api-estoque"))


class LLMFalso:
    """Substitui llm/classificador/extrator: devolve as respostas na ordem e registra cada chamada."""

    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.chamadas = []

    def invoke(self, mensagens):
        self.chamadas.append(mensagens)
        if not self.respostas:
            raise AssertionError("LLM chamado sem resposta configurada no teste")
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


@pytest.fixture(autouse=True)
def isolamento(request, monkeypatch):
    # Chamada HTTP não simulada falha (host inexistente) em vez de atingir a API real
    monkeypatch.setattr(agente, "API_URL", "http://api-de-teste.invalid")
    monkeypatch.setattr(agente, "N8N_WEBHOOK_URL", None)  # o .env local não liga o n8n nos testes
    if request.node.get_closest_marker("ollama") is None:
        for nome in ("llm", "classificador", "extrator"):
            monkeypatch.setattr(agente, nome, LLMFalso())


@pytest.fixture
def llm_falso(monkeypatch):
    """llm_falso("classificador", Decisao(...)) troca o LLM indicado e devolve o falso (para ver .chamadas)."""
    def configurar(nome, *respostas):
        falso = LLMFalso(*respostas)
        monkeypatch.setattr(agente, nome, falso)
        return falso
    return configurar


@pytest.fixture
def api_em_memoria(tmp_path, monkeypatch):
    """API de estoque real (create_app do api-estoque) sobre uma cópia temporária dos dados originais."""
    if not (API_ESTOQUE_DIR / "app" / "main.py").exists():
        pytest.skip(f"api-estoque não encontrado em {API_ESTOQUE_DIR} (defina API_ESTOQUE_DIR)")
    monkeypatch.syspath_prepend(str(API_ESTOQUE_DIR))
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.repository import CAMINHO_ORIGINAL

    dados = tmp_path / "produtos.json"
    shutil.copy(CAMINHO_ORIGINAL, dados)
    cliente = TestClient(create_app(dados))
    # O TestClient não usa timeout (e avisa se recebe); o agente sempre envia timeout=5
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None, **kw: cliente.get(url, **kw))
    monkeypatch.setattr(httpx, "post", lambda url, timeout=None, **kw: cliente.post(url, **kw))
    return cliente
