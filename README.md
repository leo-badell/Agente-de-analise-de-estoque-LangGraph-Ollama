# Agente de análise de estoque (LangGraph + Ollama)

Agente de IA que interpreta perguntas sobre estoque, consulta a [API de Estoque](https://github.com/wangsouza/api-estoque) por meio de uma ferramenta Python e classifica cada produto como **disponível**, **baixo** ou **esgotado**. O fluxo é orquestrado com LangGraph e usa um LLM local e gratuito via Ollama (`qwen2.5:7b`).

O código do agente está em [notebooks/agente_estoque.ipynb](notebooks/agente_estoque.ipynb).

## Como funciona

### Fluxo do grafo (LangGraph)

- Consulta: **pergunta → `interpretar` (LLM) → decisão → `consultar_api` (ferramenta) → `responder`**, ou direto para `responder_direto` quando a API não é necessária.
- Cadastro: **`interpretar` → `preparar_cadastro` (LLM extrai os dados) → `completar_cadastro` (pergunta o que faltar) → `confirmar_cadastro` (confirmação humana) → `responder_cadastro`**.

O diagrama completo (Mermaid) é impresso na primeira célula do notebook.

### Ferramentas: `consultar_produto` e `cadastrar_produto`

**`consultar_produto`**
- **Finalidade:** consultar um produto na API de estoque (`GET /produtos/{codigo}`).
- **Parâmetro:** `codigo` no formato `PRD-XXX`.
- **Retorno:** `{"ok": True, "produto": {...}}` ou `{"ok": False, "erro": ..., "mensagem": ...}`.
- **Falhas tratadas:** código inválido (validado antes da chamada), produto inexistente (404), serviço indisponível (503), API desligada/timeout e resposta com dados inválidos.

**`cadastrar_produto`** (altera dados: só é chamada após confirmação humana)
- **Finalidade:** cadastrar um novo produto (`POST /produtos`).
- **Parâmetro:** `produto` com `codigo`, `nome`, `categoria`, `quantidade_atual`, `estoque_minimo` e `unidade` (todos obrigatórios).
- **Retorno:** o mesmo formato da consulta.
- **Falhas tratadas:** dados inválidos (validados antes da chamada, com as mesmas regras da API), código duplicado (409), dados recusados pela API (422) e API desligada/timeout.

### Regra de situação do estoque

Calculada em Python (não pelo LLM), para o resultado ser sempre correto e verificável:

| Condição | Situação |
|---|---|
| `quantidade_atual == 0` | esgotado |
| `0 < quantidade_atual <= estoque_minimo` | baixo (inclui estar exatamente no mínimo) |
| `quantidade_atual > estoque_minimo` | disponível |

### Instruções do agente

Os prompts ficam em [agente.py](agente.py): `PROMPT_SISTEMA` (papel, objetivo, contexto e restrições), `PROMPT_CLASSIFICACAO` (intenção da mensagem, com exemplos) e `PROMPT_EXTRACAO` (dados do cadastro).

### Nós do grafo

- `interpretar`: o LLM classifica a intenção e extrai o código (saída estruturada).
- `decidir_rota`: **transição condicional** — usa a ferramenta só quando há produto a consultar.
- `consultar_api`: chama a ferramenta e calcula a situação.
- `responder`: o LLM redige a resposta com base **apenas** nos dados da API; em falha, mensagem fixa (sem LLM, para não inventar dados).
- `responder_direto`: perguntas conceituais (LLM, sem API) ou mensagens fixas para ambígua/fora do escopo.
- `preparar_cadastro`: o LLM extrai os dados do novo produto (saída estruturada). Só são aceitos valores que aparecem no texto do usuário (valor inventado ou alterado pelo modelo é descartado). Dados inválidos encerram o fluxo **antes** da confirmação. Limite: o filtro confere se o valor está no texto, não em qual campo (ex.: quantidade e mínimo trocados passam); por isso a confirmação mostra todos os valores antes do POST.
- `completar_cadastro`: se faltar algum campo, pergunta ao usuário (`interrupt()`); uma única rodada, o que continuar faltando vira erro de validação.
- `confirmar_cadastro`: **confirmação humana** com `interrupt()` do LangGraph; só chama `cadastrar_produto` (POST) se a resposta for "s".
- `responder_cadastro`: mensagem fixa com o resultado do cadastro (sem LLM, para não inventar dados).

### Execução

`perguntar()` valida a entrada (1 a 500 caracteres), executa o grafo com `recursion_limit` (parada garantida mesmo se o fluxo mudar) e mostra o estado final, que serve como evidência. No cadastro, o grafo pausa para pedir ao humano os campos que faltarem e a confirmação (s/n).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| [agente.py](agente.py) | Código do agente: ferramentas, regra de situação, prompts, nós e grafo |
| [notebooks/agente_estoque.ipynb](notebooks/agente_estoque.ipynb) | Explicação de cada parte e demonstração dos cenários (evidências) |
| [tests/](tests) | Testes automatizados com pytest |

## Como executar

Com o ambiente virtual ativo (`.venv\Scripts\Activate.ps1`):

1. Suba a API de estoque (repositório `api-estoque` na pasta ao lado):
   ```powershell
   python -m uvicorn app.main:app --app-dir ..\api-estoque --host 127.0.0.1 --port 8000
   ```
2. Deixe o Ollama rodando com o modelo `qwen2.5:7b`.
3. Abra o notebook e execute as células.

O cadastro (POST) grava no `data/produtos.json` do `api-estoque`. Para voltar aos 12 produtos originais antes de uma nova demonstração, na pasta `api-estoque`: `python scripts/reset_data.py`.

## Testes

```powershell
pytest                  # todos (inclui os cenários com o Ollama real, cerca de 1 minuto)
pytest -m "not ollama"  # só os rápidos, sem LLM
```

| Arquivo | Tipo | O que cobre |
|---|---|---|
| `tests/test_unitarios.py` | Unitário (HTTP e LLM simulados) | Regra de situação nos limites, validação do cadastro, tratamento de cada falha das ferramentas, rotas do grafo, regex do código, filtro contra valores inventados, mensagens fixas e entrada inválida |
| `tests/test_integracao.py` | Integração (API real em memória, LLM simulado) | Ferramentas contra a API (200, 404, 503, 201, 409) e o grafo completo: classificação, confirmação humana (sem "s" não há POST), pedido de campos faltantes e dados inválidos encerrando antes da confirmação |
| `tests/test_ollama.py` | Ponta a ponta (Ollama real, API em memória) | Os 3 cenários pedidos: produto existente, produto inexistente e pergunta conceitual sem chamar a API. Pulados se o Ollama estiver fora do ar |

No GitHub Actions ([.github/workflows/testes.yml](.github/workflows/testes.yml)), a cada push ou pull request: primeiro os testes rápidos; se passarem, os de ponta a ponta com o Ollama (o modelo é baixado a cada execução, cerca de 4,7 GB, e roda em CPU, então esse job leva vários minutos). O `api-estoque` é baixado do GitHub, fixado num commit.

Os testes não precisam da API no ar e **nunca alteram o `data/produtos.json` real**: cada teste usa uma cópia temporária dos dados originais (`create_app` do `api-estoque`). O caminho do `api-estoque` pode ser trocado pela variável `API_ESTOQUE_DIR` (padrão: `../api-estoque`).
