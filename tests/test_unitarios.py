"""Testes unitários: sem API e sem Ollama (HTTP e LLM simulados).

Só o que tem risco e não é coberto pela integração: limites da regra, validação antes do POST,
falhas da API que não dá para provocar na API real e as proteções contra dado inventado.
"""

import httpx
import pytest
from langchain_core.messages import AIMessage

import agente
from agente import DadosCadastro, Decisao

PRODUTO_VALIDO = {"codigo": "PRD-020", "nome": "Teclado mecânico", "categoria": "Periféricos",
                  "quantidade_atual": 15, "estoque_minimo": 5, "unidade": "un"}
PRODUTO_API = {"codigo": "PRD-001", "nome": "Mouse sem fio", "categoria": "Periféricos",
               "quantidade_atual": 5, "estoque_minimo": 10, "unidade": "un"}


def simular_http(monkeypatch, metodo, resultado):
    """Troca httpx.get/post: devolve `resultado` (Response) ou o lança (exceção). Retorna a lista de chamadas."""
    chamadas = []

    def falso(url, **kwargs):
        chamadas.append(url)
        if isinstance(resultado, Exception):
            raise resultado
        return resultado

    monkeypatch.setattr(httpx, metodo, falso)
    return chamadas


# --- Regra de situação (requisito central) ------------------------------------

@pytest.mark.parametrize("quantidade, minimo, esperado", [
    (0, 10, "esgotado"),
    (10, 10, "baixo"),        # exatamente no mínimo ainda é baixo
    (11, 10, "disponível"),
])
def test_classificar_situacao_nos_limites(quantidade, minimo, esperado):
    assert agente.classificar_situacao(quantidade, minimo) == esperado


# --- Validação antes do POST (protege a API de dados ruins) -------------------

@pytest.mark.parametrize("alteracao", [
    {"quantidade_atual": -3},
    {"quantidade_atual": "15"},   # texto não é aceito como inteiro (StrictInt)
])
def test_validar_novo_produto_recusa_dados_invalidos(alteracao):
    r = agente.validar_novo_produto({**PRODUTO_VALIDO, **alteracao})
    assert r["ok"] is False and r["erro"] == "dados_invalidos"


def test_validar_novo_produto_campo_faltando_aparece_na_mensagem():
    dados = {k: v for k, v in PRODUTO_VALIDO.items() if k != "unidade"}
    r = agente.validar_novo_produto(dados)
    assert r["ok"] is False and "unidade" in r["mensagem"]


def test_validar_novo_produto_recusa_codigo_reservado():
    r = agente.validar_novo_produto({**PRODUTO_VALIDO, "codigo": "PRD-503"})
    assert r["ok"] is False and "reservado" in r["mensagem"]


def test_cadastrar_dados_invalidos_nao_faz_post(monkeypatch):
    chamadas = simular_http(monkeypatch, "post", httpx.Response(201, json=PRODUTO_VALIDO))
    r = agente.cadastrar_produto.invoke({"produto": {**PRODUTO_VALIDO, "quantidade_atual": -3}})
    assert r["erro"] == "dados_invalidos"
    assert chamadas == []


# --- Ferramenta de consulta: entrada e falhas que a API real não simula ---------

@pytest.mark.parametrize("codigo", ["ABC-1", "PRD-0001"])
def test_consultar_codigo_invalido_nao_chama_api(monkeypatch, codigo):
    chamadas = simular_http(monkeypatch, "get", httpx.Response(200, json=PRODUTO_API))
    assert agente.consultar_produto.invoke({"codigo": codigo})["erro"] == "codigo_invalido"
    assert chamadas == []


def test_consultar_normaliza_codigo(monkeypatch):
    chamadas = simular_http(monkeypatch, "get", httpx.Response(200, json=PRODUTO_API))
    assert agente.consultar_produto.invoke({"codigo": "  prd-001 "})["ok"] is True
    assert chamadas[0].endswith("/produtos/PRD-001")


@pytest.mark.parametrize("corpo", [
    {k: v for k, v in PRODUTO_API.items() if k != "estoque_minimo"},  # campo faltando
    {**PRODUTO_API, "quantidade_atual": -1},
])
def test_consultar_resposta_200_invalida(monkeypatch, corpo):
    simular_http(monkeypatch, "get", httpx.Response(200, json=corpo))
    assert agente.consultar_produto.invoke({"codigo": "PRD-001"})["erro"] == "resposta_invalida"


def test_consultar_erro_sem_json(monkeypatch):
    simular_http(monkeypatch, "get", httpx.Response(500, text="Internal Server Error"))
    r = agente.consultar_produto.invoke({"codigo": "PRD-001"})
    assert r == {"ok": False, "erro": "http_500", "mensagem": "Erro inesperado da API."}


@pytest.mark.parametrize("metodo, ferramenta, argumento", [
    ("get", "consultar_produto", {"codigo": "PRD-001"}),
    ("post", "cadastrar_produto", {"produto": PRODUTO_VALIDO}),
])
def test_api_desligada(monkeypatch, metodo, ferramenta, argumento):
    simular_http(monkeypatch, metodo, httpx.ConnectError("recusada"))
    assert getattr(agente, ferramenta).invoke(argumento) == agente.API_INACESSIVEL


# --- Interpretação da pergunta ------------------------------------------------

@pytest.mark.parametrize("texto, esperado", [
    ("Como está o PRD-001?", "PRD-001"),
    ("tem o PRD-0011?", None),    # não pode virar consulta ao PRD-001
])
def test_regex_codigo_no_texto(texto, esperado):
    achado = agente.CODIGO_NO_TEXTO.search(texto)
    assert (achado.group(0) if achado else None) == esperado


def test_interpretar_regex_tem_prioridade_sobre_codigo_do_llm(llm_falso):
    llm_falso("classificador", Decisao(intencao="consultar_produto", codigo="PRD-01"))
    assert agente.interpretar({"pergunta": "Ainda tem prd-004?"}) == {
        "intencao": "consultar_produto", "codigo": "prd-004"}


def test_interpretar_falha_do_llm(llm_falso):
    llm_falso("classificador", ConnectionError("Ollama fora do ar"))
    assert agente.interpretar({"pergunta": "Tem PRD-001?"}) == {"intencao": "falha_llm"}


def test_consulta_sem_codigo_pede_codigo_sem_chamar_api():
    estado = {"intencao": "consultar_produto", "codigo": None}
    assert agente.decidir_rota(estado) == "responder_direto"
    assert agente.responder_direto(estado) == {"resposta": agente.MENSAGENS_FIXAS["ambigua"]}


# --- Proteções contra dado inventado pelo LLM ---------------------------------

def test_extrair_dados_descarta_valor_inventado(llm_falso):
    texto = ("Cadastre o PRD-020: nome Teclado mecânico, categoria Periféricos, "
             "quantidade atual 15, estoque mínimo 5, unidade un")
    llm_falso("extrator", DadosCadastro(**{**PRODUTO_VALIDO, "quantidade_atual": 99, "nome": "Mouse"}))
    dados = agente.extrair_dados(texto)
    assert "quantidade_atual" not in dados and "nome" not in dados


def test_responder_sempre_inclui_quantidade_e_minimo(llm_falso):
    # Quantidade 0: o "0" de "PRD-004" e de "20" não pode contar como quantidade citada
    llm_falso("llm", AIMessage(content="O Cabo HDMI 2 m (PRD-004) está esgotado. Estoque mínimo: 20."))
    produto = {**PRODUTO_API, "codigo": "PRD-004", "nome": "Cabo HDMI 2 m", "quantidade_atual": 0, "estoque_minimo": 20}
    estado = {"pergunta": "Tem PRD-004?", "resultado": {"ok": True, "produto": produto}, "situacao": "esgotado"}
    assert "0 un em estoque, estoque mínimo 20, situação: esgotado" in agente.responder(estado)["resposta"]


def test_responder_usa_resumo_fixo_se_llm_falhar(llm_falso):
    llm_falso("llm", ConnectionError("Ollama fora do ar"))
    estado = {"pergunta": "Tem PRD-001?", "resultado": {"ok": True, "produto": PRODUTO_API}, "situacao": "baixo"}
    assert agente.responder(estado)["resposta"] == (
        "Mouse sem fio (PRD-001): 5 un em estoque, estoque mínimo 10, situação: baixo.")


# --- Entrada do usuário -------------------------------------------------------

@pytest.mark.parametrize("pergunta", ["   ", "x" * 501])
def test_perguntar_recusa_entrada_invalida(pergunta):
    assert agente.perguntar(pergunta) == {}
    assert agente.classificador.chamadas == []


# --- Integração com o n8n -----------------------------------------------------

ESTADO_BAIXO = {"pergunta": "Tem PRD-001?", "intencao": "consultar_produto", "codigo": "PRD-001",
                "situacao": "baixo", "resposta": "..."}


def test_n8n_desligado_sem_url_nao_faz_post(monkeypatch):
    chamadas = simular_http(monkeypatch, "post", httpx.Response(200, json={"alerta": True}))
    assert agente.notificar_n8n(ESTADO_BAIXO) is None
    assert chamadas == []


def test_n8n_envia_resultado_e_devolve_alerta(monkeypatch):
    monkeypatch.setattr(agente, "N8N_WEBHOOK_URL", "http://n8n-de-teste.invalid/webhook/estoque-resultado")
    enviados = []
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: enviados.append(json) or
                        httpx.Response(200, json={"alerta": True, "mensagem": "Repor o produto PRD-001"}))
    assert agente.notificar_n8n(ESTADO_BAIXO) == {"alerta": True, "mensagem": "Repor o produto PRD-001"}
    assert enviados == [ESTADO_BAIXO]


@pytest.mark.parametrize("resultado", [httpx.ConnectError("recusada"), httpx.Response(404, text="not found")])
def test_n8n_fora_do_ar_nao_interrompe_agente(monkeypatch, resultado):
    monkeypatch.setattr(agente, "N8N_WEBHOOK_URL", "http://n8n-de-teste.invalid/webhook/estoque-resultado")
    simular_http(monkeypatch, "post", resultado)
    assert agente.notificar_n8n(ESTADO_BAIXO) is None
