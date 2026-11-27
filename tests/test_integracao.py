"""Testes de integração: API de estoque real em memória (cópia temporária dos dados) + LLM simulado.

Grafo completo de ponta a ponta, com foco no que tem risco: confirmação humana antes do POST,
dados inválidos barrados, falhas da API sem resposta inventada e perguntas que não devem chamar a API.
"""

import builtins

import pytest
from langchain_core.messages import AIMessage

import agente
from agente import DadosCadastro, Decisao

NOVO = {"codigo": "PRD-020", "nome": "Teclado mecânico", "categoria": "Periféricos",
        "quantidade_atual": 15, "estoque_minimo": 5, "unidade": "un"}
PEDIDO_CADASTRO = ("Cadastre o produto PRD-020: nome Teclado mecânico, categoria Periféricos, "
                   "quantidade atual 15, estoque mínimo 5, unidade un")


@pytest.fixture
def sem_input(monkeypatch):
    """Falha se o agente pedir algo ao humano que o teste não previu."""
    def falha(pedido=""):
        raise AssertionError(f"pedido inesperado ao humano: {pedido}")
    monkeypatch.setattr(builtins, "input", falha)


def existe_na_api(cliente, codigo):
    return cliente.get(f"/produtos/{codigo}").status_code == 200


def configurar_cadastro(llm_falso, produto):
    llm_falso("classificador", Decisao(intencao="cadastrar_produto", codigo=produto["codigo"]))
    llm_falso("extrator", DadosCadastro(**produto))


# --- Consulta -----------------------------------------------------------------

def test_consulta_classifica_com_dados_da_api(api_em_memoria, llm_falso):
    llm_falso("classificador", Decisao(intencao="consultar_produto", codigo="PRD-005"))
    llm_falso("llm", AIMessage(content="Resposta redigida pelo LLM."))
    estado = agente.perguntar("Como está o PRD-005?")  # 30 unidades, mínimo 30
    assert estado["resultado"]["ok"] is True
    assert estado["situacao"] == "baixo"


def test_produto_inexistente_responde_sem_llm(api_em_memoria, llm_falso):
    llm_falso("classificador", Decisao(intencao="consultar_produto", codigo="PRD-999"))
    estado = agente.perguntar("Qual o estoque do PRD-999?")
    assert estado["resultado"]["erro"] == "produto_nao_encontrado"
    assert estado["resposta"].startswith("Não consegui obter os dados do produto PRD-999")
    assert agente.llm.chamadas == []  # sem LLM: não há como inventar dados


def test_api_indisponivel_503(api_em_memoria):
    r = agente.consultar_produto.invoke({"codigo": "PRD-503"})
    assert r["ok"] is False and r["erro"] == "servico_indisponivel"


def test_conceitual_nao_consulta_api(api_em_memoria, llm_falso, monkeypatch):
    llm_falso("classificador", Decisao(intencao="conceitual"))
    llm_falso("llm", AIMessage(content="Estoque mínimo é o ponto de reposição."))
    monkeypatch.setattr(agente.httpx, "get", lambda *a, **k: pytest.fail("a API não deveria ser chamada"))
    estado = agente.perguntar("O que é estoque mínimo?")
    assert estado["resposta"] == "Estoque mínimo é o ponto de reposição."
    assert "resultado" not in estado


# --- Cadastro (POST) ----------------------------------------------------------

def test_cadastro_confirmado_faz_post(api_em_memoria, llm_falso):
    configurar_cadastro(llm_falso, NOVO)
    estado = agente.perguntar(PEDIDO_CADASTRO, respostas=["s"])
    assert estado["resultado"] == {"ok": True, "produto": NOVO}
    assert existe_na_api(api_em_memoria, "PRD-020")


@pytest.mark.parametrize("resposta", ["n", ""])
def test_cadastro_sem_confirmacao_nao_faz_post(api_em_memoria, llm_falso, resposta):
    configurar_cadastro(llm_falso, NOVO)
    estado = agente.perguntar(PEDIDO_CADASTRO, respostas=[resposta])
    assert estado["resultado"]["erro"] == "cancelado"
    assert not existe_na_api(api_em_memoria, "PRD-020")


def test_cadastro_pede_campos_faltantes(api_em_memoria, llm_falso):
    llm_falso("classificador", Decisao(intencao="cadastrar_produto", codigo="PRD-030"))
    llm_falso("extrator",
              DadosCadastro(codigo="PRD-030", nome="Webcam HD", categoria="Periféricos", quantidade_atual=8),
              DadosCadastro(estoque_minimo=3, unidade="un"))
    estado = agente.perguntar("Quero cadastrar o PRD-030, uma Webcam HD da categoria Periféricos, com 8 unidades",
                              respostas=["estoque mínimo 3, unidade un", "s"])
    assert estado["resultado"]["ok"] is True
    assert existe_na_api(api_em_memoria, "PRD-030")


def test_cadastro_invalido_encerra_antes_da_confirmacao(api_em_memoria, llm_falso, sem_input):
    configurar_cadastro(llm_falso, {**NOVO, "codigo": "PRD-021", "quantidade_atual": -3})
    estado = agente.perguntar(PEDIDO_CADASTRO.replace("PRD-020", "PRD-021").replace("15", "-3"))
    assert estado["resultado"]["erro"] == "dados_invalidos"
    assert not existe_na_api(api_em_memoria, "PRD-021")


def test_cadastro_duplicado(api_em_memoria, llm_falso):
    configurar_cadastro(llm_falso, {**NOVO, "codigo": "PRD-001"})
    estado = agente.perguntar(PEDIDO_CADASTRO.replace("PRD-020", "PRD-001"), respostas=["s"])
    assert estado["resultado"]["erro"] == "produto_duplicado"
    assert estado["resposta"].startswith("Não foi possível cadastrar o produto")
