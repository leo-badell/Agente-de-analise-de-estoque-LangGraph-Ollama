"""Agente de análise de estoque (LangGraph + Ollama). Demonstração em notebooks/agente_estoque.ipynb."""

import os
import re
import uuid
from typing import Literal, Optional, TypedDict

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

load_dotenv()  # lê o .env, se existir (ele não é versionado)
API_URL = os.getenv("API_ESTOQUE_URL", "http://127.0.0.1:8000")
MODELO = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# temperature=0 deixa as respostas estáveis; timeout evita ficar preso se o Ollama travar
llm = ChatOllama(model=MODELO, temperature=0, client_kwargs={"timeout": 120})


CODIGO_VALIDO = re.compile(r"^PRD-[0-9]{3}$")
CAMPOS_OBRIGATORIOS = {"codigo", "nome", "quantidade_atual", "estoque_minimo"}
API_INACESSIVEL = {"ok": False, "erro": "api_inacessivel",
                   "mensagem": "Não foi possível conectar à API de estoque. Verifique se ela está no ar."}


def _corpo_json(r: httpx.Response) -> dict:
    try:
        corpo = r.json()
    except ValueError:
        return {}
    return corpo if isinstance(corpo, dict) else {}


@tool
def consultar_produto(codigo: str) -> dict:
    """Consulta um produto na API de estoque pelo código (formato PRD-XXX, ex.: PRD-001).

    Retorna {"ok": True, "produto": {...}} ou {"ok": False, "erro": <identificador>, "mensagem": <texto>}.
    """
    codigo = codigo.strip().upper()
    if not CODIGO_VALIDO.match(codigo):
        return {"ok": False, "erro": "codigo_invalido",
                "mensagem": f"O código '{codigo}' não segue o formato PRD-XXX (ex.: PRD-001)."}

    try:
        r = httpx.get(f"{API_URL}/produtos/{codigo}", timeout=5)
    except httpx.HTTPError:
        return API_INACESSIVEL

    corpo = _corpo_json(r)
    if r.status_code != 200:
        return {"ok": False, "erro": corpo.get("erro", f"http_{r.status_code}"),
                "mensagem": corpo.get("mensagem", "Erro inesperado da API.")}

    # Verifica o retorno antes de usar: campos presentes e quantidades inteiras >= 0
    quantidades_ok = all(isinstance(corpo.get(c), int) and corpo[c] >= 0
                         for c in ("quantidade_atual", "estoque_minimo"))
    if not (CAMPOS_OBRIGATORIOS <= corpo.keys() and quantidades_ok):
        return {"ok": False, "erro": "resposta_invalida",
                "mensagem": "A API retornou dados incompletos ou inválidos."}
    return {"ok": True, "produto": corpo}


class NovoProduto(BaseModel):
    """Corpo do POST /produtos, com as mesmas regras da API (todos os campos obrigatórios)."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    codigo: str = Field(pattern=CODIGO_VALIDO.pattern)
    nome: str = Field(min_length=1)
    categoria: str = Field(min_length=1)
    quantidade_atual: StrictInt = Field(ge=0)
    estoque_minimo: StrictInt = Field(ge=0)
    unidade: str = Field(min_length=1)


def validar_novo_produto(dados: dict) -> dict:
    """Retorna {"ok": True, "produto": {...}} ou {"ok": False, "erro": "dados_invalidos", "mensagem": ...}."""
    try:
        produto = NovoProduto.model_validate(dados).model_dump()
    except ValidationError as e:
        problemas = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        return {"ok": False, "erro": "dados_invalidos", "mensagem": f"Dados inválidos para cadastro ({problemas})."}
    if produto["codigo"] == "PRD-503":
        return {"ok": False, "erro": "dados_invalidos",
                "mensagem": "O código PRD-503 é reservado pela API e não pode ser cadastrado."}
    return {"ok": True, "produto": produto}


@tool
def cadastrar_produto(produto: dict) -> dict:
    """Cadastra um novo produto na API de estoque (POST /produtos). Use somente após confirmação humana.

    produto: codigo (PRD-XXX), nome, categoria, quantidade_atual (inteiro >= 0), estoque_minimo (inteiro >= 0), unidade.
    Retorna {"ok": True, "produto": {...}} ou {"ok": False, "erro": <identificador>, "mensagem": <texto>}.
    """
    validacao = validar_novo_produto(produto)
    if not validacao["ok"]:
        return validacao

    try:
        r = httpx.post(f"{API_URL}/produtos", json=validacao["produto"], timeout=5)
    except httpx.HTTPError:
        return API_INACESSIVEL

    corpo = _corpo_json(r)
    if r.status_code != 201:  # 409 produto_duplicado, 422 dados_invalidos
        return {"ok": False, "erro": corpo.get("erro", f"http_{r.status_code}"),
                "mensagem": corpo.get("mensagem", "Erro inesperado da API.")}
    return {"ok": True, "produto": corpo}


def classificar_situacao(quantidade: int, minimo: int) -> str:
    if quantidade == 0:
        return "esgotado"
    if quantidade <= minimo:
        return "baixo"
    return "disponível"


PROMPT_SISTEMA = """Papel: você é um assistente de análise de estoque de uma loja.
Objetivo: informar a disponibilidade e a situação do estoque de produtos.
Contexto: os dados vêm exclusivamente da API de estoque. Cada produto tem código PRD-XXX,
quantidade_atual e estoque_minimo. Situações: esgotado (quantidade 0), baixo (quantidade maior
que 0 e menor ou igual ao mínimo) e disponível (acima do mínimo).
O estoque mínimo é o ponto de reposição: quando a quantidade atual chega ao mínimo ou fica abaixo dele,
o produto precisa ser reposto para não esgotar.
Restrições:
- Use somente os dados fornecidos; nunca invente produtos, códigos ou quantidades.
- Você consulta produtos e cadastra novos somente após confirmação humana; não altera nem exclui produtos.
- Ao falar de um produto, informe sempre a quantidade atual e o estoque mínimo.
- Responda em português, de forma breve e objetiva."""

PROMPT_CLASSIFICACAO = """Classifique a mensagem do usuário em uma intenção:
- consultar_produto: pede estoque, quantidade, disponibilidade ou situação de um produto e informa um código
  (copie o código exatamente como o usuário escreveu no campo codigo, mesmo que pareça errado).
- conceitual: pergunta geral sobre conceitos de estoque (ex.: o que é estoque mínimo), sem pedir dados de um produto.
- ambigua: quer dados de produto, mas não informou nenhum código (nome de produto, como mouse ou teclado, não é código).
- cadastrar_produto: pede para cadastrar, incluir ou criar um novo produto.
- fora_do_escopo: qualquer outro assunto, inclusive pedidos para alterar ou excluir produtos.

Exemplos:
- "Qual a situação do PRD-001?" -> consultar_produto, codigo PRD-001
- "Tem o item ABC-1 em estoque?" -> consultar_produto, codigo ABC-1
- "O que significa estoque baixo?" -> conceitual
- "Como está o estoque do mouse?" -> ambigua
- "Cadastre o produto PRD-020, Teclado, categoria Periféricos, 15 unidades, mínimo 5" -> cadastrar_produto, codigo PRD-020
- "Exclua o produto PRD-001" -> fora_do_escopo
- "Altere a quantidade do PRD-002 para 30" -> fora_do_escopo
- "Qual a capital da França?" -> fora_do_escopo"""

PROMPT_EXTRACAO = """Extraia os dados do produto que o usuário quer cadastrar.
Campos: codigo, nome, categoria, quantidade_atual, estoque_minimo, unidade.
- Copie os valores exatamente como o usuário escreveu, inclusive números negativos.
- Nunca invente: deixe nulo o campo que o usuário não informou."""


class Decisao(BaseModel):
    intencao: Literal["consultar_produto", "cadastrar_produto", "conceitual", "ambigua", "fora_do_escopo"]
    codigo: Optional[str] = Field(None, description="Código do produto citado, exatamente como escrito pelo usuário")


class DadosCadastro(BaseModel):
    """Saída do LLM no cadastro: campo não informado fica nulo (a validação acusa a falta)."""
    codigo: Optional[str] = None
    nome: Optional[str] = None
    categoria: Optional[str] = None
    quantidade_atual: Optional[int] = None
    estoque_minimo: Optional[int] = None
    unidade: Optional[str] = None


class Estado(TypedDict, total=False):
    pergunta: str
    intencao: str
    codigo: Optional[str]
    dados_cadastro: dict     # produto extraído da pergunta, aguardando confirmação
    resultado: dict          # retorno da ferramenta
    situacao: Optional[str]
    resposta: str


MENSAGENS_FIXAS = {
    "ambigua": "Não identifiquei qual produto você quer consultar. Informe o código no formato PRD-XXX (ex.: PRD-001).",
    "fora_do_escopo": "Só posso consultar o estoque e cadastrar novos produtos (com sua confirmação). "
                      "Não altero nem excluo produtos.",
    "falha_llm": "Não consegui processar a solicitação agora (modelo local indisponível). "
                 "Verifique se o Ollama está rodando e tente novamente.",
}

classificador = llm.with_structured_output(Decisao)
extrator = llm.with_structured_output(DadosCadastro)


CODIGO_NO_TEXTO = re.compile(r"\bPRD-[0-9]{3}\b", re.IGNORECASE)


def interpretar(estado: Estado) -> Estado:
    try:
        d = classificador.invoke([("system", PROMPT_CLASSIFICACAO), ("human", estado["pergunta"])])
        # O LLM decide a intenção; o código PRD-XXX é extraído por regex (modelos pequenos às vezes o omitem)
        achado = CODIGO_NO_TEXTO.search(estado["pergunta"])
        return {"intencao": d.intencao, "codigo": achado.group(0) if achado else d.codigo}
    except Exception as e:  # Ollama fora do ar, timeout ou saída fora do formato
        print(f"[interpretar] falha no LLM: {e}")
        return {"intencao": "falha_llm"}


def decidir_rota(estado: Estado) -> str:
    if estado["intencao"] == "consultar_produto" and estado.get("codigo"):
        return "consultar_api"
    if estado["intencao"] == "cadastrar_produto":
        return "preparar_cadastro"
    return "responder_direto"


def consultar_api(estado: Estado) -> Estado:
    resultado = consultar_produto.invoke({"codigo": estado["codigo"]})
    situacao = None
    if resultado["ok"]:
        p = resultado["produto"]
        situacao = classificar_situacao(p["quantidade_atual"], p["estoque_minimo"])
    return {"resultado": resultado, "situacao": situacao}


def responder(estado: Estado) -> Estado:
    r = estado["resultado"]
    if not r["ok"]:
        return {"resposta": f"Não consegui obter os dados do produto {estado['codigo']}: {r['mensagem']}"}

    p = r["produto"]
    resumo = (f"{p['nome']} ({p['codigo']}): {p['quantidade_atual']} {p.get('unidade', '')} em estoque, "
              f"estoque mínimo {p['estoque_minimo']}, situação: {estado['situacao']}.")
    try:
        msg = llm.invoke([
            ("system", PROMPT_SISTEMA),
            ("human", f"Pergunta: {estado['pergunta']}\nDados da API: {resumo}\n"
                      "Responda à pergunta usando apenas esses dados e cite a quantidade atual e o estoque mínimo."),
        ])
        # Resumo sempre no fim: quantidade atual e mínimo garantidos mesmo se o modelo os omitir
        return {"resposta": f"{msg.content}\n{resumo}"}
    except Exception as e:
        print(f"[responder] falha no LLM, usando resumo fixo: {e}")
        return {"resposta": resumo}


def responder_direto(estado: Estado) -> Estado:
    if estado["intencao"] == "conceitual":
        try:
            msg = llm.invoke([
                ("system", PROMPT_SISTEMA + "\nResponda de forma conceitual, sem citar produtos nem quantidades."),
                ("human", estado["pergunta"]),
            ])
            return {"resposta": msg.content}
        except Exception as e:
            print(f"[responder_direto] falha no LLM: {e}")
            return {"resposta": MENSAGENS_FIXAS["falha_llm"]}
    # consultar_produto sem código cai aqui e é tratado como ambígua
    return {"resposta": MENSAGENS_FIXAS.get(estado["intencao"], MENSAGENS_FIXAS["ambigua"])}


CAMPOS_CADASTRO = list(NovoProduto.model_fields)
FALHA_LLM_CADASTRO = {"resultado": {"ok": False, "erro": "falha_llm", "mensagem": MENSAGENS_FIXAS["falha_llm"]}}


def extrair_dados(texto: str) -> dict:
    """O LLM extrai os campos; só ficam os valores que aparecem no texto (evita dado inventado ou alterado)."""
    d = extrator.invoke([("system", PROMPT_EXTRACAO), ("human", texto)])
    numeros = set(re.findall(r"-?\d+", texto))
    dados = {}
    for campo, valor in d.model_dump().items():
        if valor is None:
            continue
        presente = str(valor) in numeros if isinstance(valor, int) else str(valor).strip().lower() in texto.lower()
        if presente:
            dados[campo] = valor
    return dados


def faltando(dados: dict) -> list:
    return [c for c in CAMPOS_CADASTRO if c not in dados]


def validar_cadastro(dados: dict) -> Estado:
    validacao = validar_novo_produto(dados)
    if not validacao["ok"]:
        return {"dados_cadastro": dados, "resultado": validacao}
    return {"dados_cadastro": validacao["produto"]}


def preparar_cadastro(estado: Estado) -> Estado:
    try:
        dados = extrair_dados(estado["pergunta"])
    except Exception as e:
        print(f"[preparar_cadastro] falha no LLM: {e}")
        return FALHA_LLM_CADASTRO
    if estado.get("codigo"):  # código copiado do texto pela regex, como na consulta
        dados["codigo"] = estado["codigo"].upper()
    if faltando(dados):
        return {"dados_cadastro": dados}  # completar_cadastro pergunta o que falta
    return validar_cadastro(dados)


def completar_cadastro(estado: Estado) -> Estado:
    dados = dict(estado["dados_cadastro"])
    faltam = faltando(dados)
    resposta = interrupt(f"Para cadastrar, informe: {', '.join(faltam)} (ex.: estoque mínimo 5, unidade un).")
    try:
        novos = extrair_dados(str(resposta))
    except Exception as e:
        print(f"[completar_cadastro] falha no LLM: {e}")
        return FALHA_LLM_CADASTRO
    dados.update({c: v for c, v in novos.items() if c in faltam})
    return validar_cadastro(dados)  # rodada única: o que ainda faltar vira erro de validação


def decidir_cadastro(estado: Estado) -> str:
    if "resultado" in estado:  # erro (dados inválidos ou falha do LLM): encerra sem pedir confirmação
        return "responder_cadastro"
    if faltando(estado["dados_cadastro"]):
        return "completar_cadastro"
    return "confirmar_cadastro"


def confirmar_cadastro(estado: Estado) -> Estado:
    p = estado["dados_cadastro"]
    resposta = interrupt(f"Confirma o cadastro de {p['codigo']} - {p['nome']} ({p['categoria']}), "
                         f"{p['quantidade_atual']} {p['unidade']}, estoque mínimo {p['estoque_minimo']}? (s/n)")
    # Ao retomar, o nó roda de novo e interrupt() devolve a resposta: o POST só acontece depois dela
    if str(resposta).strip().lower() not in ("s", "sim"):
        return {"resultado": {"ok": False, "erro": "cancelado", "mensagem": "cadastro cancelado pelo usuário."}}
    return {"resultado": cadastrar_produto.invoke({"produto": p})}


def responder_cadastro(estado: Estado) -> Estado:
    r = estado["resultado"]
    if not r["ok"]:
        return {"resposta": f"Não foi possível cadastrar o produto: {r['mensagem']}"}
    p = r["produto"]
    situacao = classificar_situacao(p["quantidade_atual"], p["estoque_minimo"])
    return {"situacao": situacao,
            "resposta": f"Produto {p['codigo']} ({p['nome']}) cadastrado com sucesso: {p['quantidade_atual']} "
                        f"{p['unidade']} em estoque, estoque mínimo {p['estoque_minimo']}, situação: {situacao}."}


grafo = StateGraph(Estado)
grafo.add_node("interpretar", interpretar)
grafo.add_node("consultar_api", consultar_api)
grafo.add_node("responder", responder)
grafo.add_node("responder_direto", responder_direto)
grafo.add_node("preparar_cadastro", preparar_cadastro)
grafo.add_node("completar_cadastro", completar_cadastro)
grafo.add_node("confirmar_cadastro", confirmar_cadastro)
grafo.add_node("responder_cadastro", responder_cadastro)

grafo.add_edge(START, "interpretar")
grafo.add_conditional_edges("interpretar", decidir_rota, ["consultar_api", "preparar_cadastro", "responder_direto"])
ROTAS_CADASTRO = ["completar_cadastro", "confirmar_cadastro", "responder_cadastro"]
grafo.add_conditional_edges("preparar_cadastro", decidir_cadastro, ROTAS_CADASTRO)
grafo.add_conditional_edges("completar_cadastro", decidir_cadastro, ROTAS_CADASTRO[1:])  # rodada única, sem laço
grafo.add_edge("confirmar_cadastro", "responder_cadastro")
grafo.add_edge("consultar_api", "responder")
grafo.add_edge("responder", END)          # condição de encerramento
grafo.add_edge("responder_direto", END)   # condição de encerramento
grafo.add_edge("responder_cadastro", END) # condição de encerramento

agente = grafo.compile(checkpointer=InMemorySaver())  # o checkpointer permite pausar na confirmação humana


LIMITE_PERGUNTA = 500
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")  # vazia = integração com o n8n desligada


def notificar_n8n(estado: Estado) -> Optional[dict]:
    """Envia o resultado da análise ao workflow do n8n e devolve a resposta dele (alerta de reposição).

    Falha do n8n não interrompe o agente: só avisa e retorna None.
    """
    if not N8N_WEBHOOK_URL:
        return None
    dados = {c: estado.get(c) for c in ("pergunta", "intencao", "codigo", "situacao", "resposta")}
    try:
        r = httpx.post(N8N_WEBHOOK_URL, json=dados, timeout=5)
    except httpx.HTTPError as e:
        print(f"[n8n] não foi possível notificar o workflow: {e}")
        return None
    corpo = _corpo_json(r)
    if r.status_code != 200 or not corpo:
        print(f"[n8n] resposta inesperada do workflow (HTTP {r.status_code}).")
        return None
    return corpo


def perguntar(pergunta: str, respostas: Optional[list] = None) -> Estado:
    pergunta = (pergunta or "").strip()
    if not pergunta or len(pergunta) > LIMITE_PERGUNTA:
        print(f"Entrada inválida: a pergunta deve ter entre 1 e {LIMITE_PERGUNTA} caracteres.")
        return {}

    config = {"recursion_limit": 10, "configurable": {"thread_id": str(uuid.uuid4())}}
    estado = agente.invoke({"pergunta": pergunta}, config=config)
    respostas = list(respostas or [])
    while estado.get("__interrupt__"):  # o agente pede algo ao humano (campos faltantes ou confirmação)
        pedido = estado["__interrupt__"][0].value
        resposta = respostas.pop(0) if respostas else input(pedido + " ")
        print(f"Agente   : {pedido}\nHumano   : {resposta}")
        estado = agente.invoke(Command(resume=resposta), config=config)
    print(f"Pergunta : {pergunta}")
    print(f"Intenção : {estado.get('intencao')} | código: {estado.get('codigo')} | situação: {estado.get('situacao')}")
    print(f"API      : {estado.get('resultado', 'não consultada')}")
    print(f"Resposta : {estado['resposta']}\n")
    retorno = notificar_n8n(estado)
    if retorno:
        print(f"n8n      : {retorno}\n")
    return estado
