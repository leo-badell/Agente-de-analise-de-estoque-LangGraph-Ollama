"""Cenários de teste do PDF com o LLM real (Ollama) e a API em memória.

Lentos e dependentes do modelo: as verificações olham o estado do grafo, não o texto exato da resposta.
Pulados automaticamente se o Ollama não estiver no ar com o modelo configurado.
Para rodar só os testes rápidos: pytest -m "not ollama"
"""

import os

import httpx
import pytest

import agente


def _ollama_pronto() -> bool:
    url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        modelos = httpx.get(f"{url}/api/tags", timeout=3).json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return False
    return any(m.get("name") == agente.MODELO for m in modelos)


pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(not _ollama_pronto(), reason=f"Ollama fora do ar ou sem o modelo {agente.MODELO}"),
]


def test_cenario_1_produto_existente(api_em_memoria):
    estado = agente.perguntar("Qual a situação do estoque do produto PRD-001?")
    assert estado["intencao"] == "consultar_produto"
    assert estado["codigo"].upper() == "PRD-001"
    assert estado["resultado"]["ok"] is True
    assert estado["situacao"] == "baixo"
    assert "5" in estado["resposta"]  # quantidade atual sempre presente na resposta


def test_cenario_2_produto_inexistente(api_em_memoria):
    estado = agente.perguntar("Qual o estoque do PRD-999?")
    assert estado["intencao"] == "consultar_produto"
    assert estado["resultado"]["ok"] is False
    assert estado["resultado"]["erro"] == "produto_nao_encontrado"
    assert "PRD-999" in estado["resposta"]


def test_cenario_3_pergunta_conceitual_sem_api(api_em_memoria, monkeypatch):
    monkeypatch.setattr(agente.httpx, "get", lambda *a, **k: pytest.fail("a API não deveria ser chamada"))
    estado = agente.perguntar("O que é estoque mínimo?")
    assert estado["intencao"] == "conceitual"
    assert "resultado" not in estado
    assert estado["resposta"].strip()
