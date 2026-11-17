# Base de moderação de apelidos — ByteCraft

Protótipo local e experimental de moderação de apelidos para um jogo educacional destinado a estudantes do 5º ano. A integração e as regras lexicais são experimentais e não substituem validação independente nem revisão apropriada antes de uso em produção.

## Estrutura

- `data/raw/`: arquivos locais originais do LDNOOBW e OLID-BR. Não editar nem sobrescrever.
- `data/interim/`: reservado para transformações rastreáveis, ainda sem dados derivados.
- `data/curated/nickname_examples.csv`: exemplos sintéticos e candidatos LDNOOBW com rótulos sugeridos e status de revisão.
- `data/curated/manual_evaluation.csv`: casos manuais revisados, usados no treinamento e na regra lexical.
- `data/curated/bytecraft_nicknames_moderacao.csv`: exemplos revisados adicionados pelo ByteCraft para treino e regra lexical.
- `data/curated/ldnoobw_candidate_reviews.csv`: decisões humanas dos 76 termos-base LDNOOBW, referenciadas pelo número de linha.
- `data/processed/`: reservado para dados preparados após aprovação da política; vazio nesta etapa.
- `docs/labeling_policy.md`: critérios de classificação e revisão.
- `docs/olid_br_candidate_plan.md`: plano futuro para analisar candidatos do OLID-BR sem converter comentários em apelidos.
- `src/nickname_normalization.py`: transformações de comparação não destrutivas.
- `src/lexical_filter.py`: consulta explicável de candidatos LDNOOBW; não bloqueia automaticamente.
- `src/review_nicknames.py`: revisão humana explícita por terminal, com gravação atômica.
- `src/nickname_dataset.py`: carregamento, validação e resumo da base autoral.
- `src/nickname_model.py`: auditoria, preparação, pipeline experimental, avaliação agrupada e previsão cautelosa.
- `src/predict_nickname.py`: moderador interativo no terminal; carrega o artefato já treinado e não repete o treinamento.
- `models/nickname_moderation.joblib`: pipeline salvo, incluindo vetorizador e classificador (gerado pelo treinamento).
- `models/nickname_moderation.metadata.json`: hash e contagens da base, parâmetros, versões das bibliotecas, estratégia de divisão e métricas.
- `tests/test_nickname_dataset.py`: testes automatizados da ferramenta.
- `tests/test_nickname_model.py`: testes da elegibilidade, preparação, divisão, treinamento, persistência e previsão.
- `tests/test_predict_nickname.py`: testes do fluxo interativo, entradas vazias, incerteza e encerramento.
- `tests/test_review_nicknames.py`: testes de revisão explícita, preservação e erro seguro.
- `reports/data_diagnostic.md`: diagnóstico dos arquivos locais e pendências de procedência.
- `reports/training_readiness.md`: auditoria de elegibilidade da etapa de treinamento.

## Dados oficiais e exemplos autorais

O conteúdo em `data/raw/` é a cópia local dos datasets existentes no computador. O OLID-BR está acompanhado de README que declara a versão v1.0, licença CC-BY-4.0 e citação bibliográfica; ainda falta confirmar que os arquivos locais correspondem exatamente à distribuição oficial. A origem e a licença do arquivo LDNOOBW local não estão documentadas junto ao arquivo e permanecem pendentes. Não se afirma que qualquer licença tenha sido verificada online.

`data/curated/nickname_examples.csv` contém exemplos sintéticos fictícios e termos do LDNOOBW revisados pelo responsável. As 76 entradas originais estão representadas sem duplicar os sete termos que já constavam na base. Cada entrada ganhou duas variações (sufixo numérico e substituição/inserção de símbolo), agrupadas em `root_group` com a forma-base e mantidas como sugestões com `review_status=pendente`. Nenhuma variação pendente entra no treinamento. Nenhum exemplo representa crianças reais.

`nickname_examples.csv` tem 359 registros: 69 `permitido` e 290 `inadequado`; 207 estão revisados e 152 são variações LDNOOBW pendentes. O modelo foi retreinado com 371 exemplos elegíveis após deduplicação: 138 sintéticos, 69 LDNOOBW, 14 de avaliação manual e 150 da base ByteCraft. As variações pendentes estão excluídas. O split agrupado reteve 74 exemplos para teste (9 permitidos e 65 inadequados), com acurácia 0,838 e macro-F1 0,680. A decisão automática bloqueou 27 inadequados corretamente e 2 permitidos incorretamente, deixando 38 inadequados sem bloqueio; o desempenho é experimental e não deve ser interpretado como validação de produção. As contagens finais e hashes das fontes ficam no metadata do modelo.

Foram reparados apenas artefatos textuais decorrentes de substituição indevida em descrições (por exemplo, `dependente` escrito como `derevisado`); nenhum rótulo nem status foi alterado nessa correção.

## Validação

No PowerShell, primeiro entre na pasta `nickname-moderation`. Se você estiver na raiz do repositório ByteCraft, use:

```powershell
Set-Location .\nickname-moderation
```

Se o projeto ainda não tiver um ambiente virtual, crie-o e ative-o:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas scikit-learn joblib
```

Se o ambiente virtual já existir, ative-o em vez de criar outro. Depois, execute:

```powershell
& ..\.venv\Scripts\python.exe -m unittest discover -s tests -v
& ..\.venv\Scripts\python.exe -m src.nickname_dataset
```

Esses comandos pressupõem que o ambiente virtual compartilhado está em `ByteCraft\.venv`. Se o ambiente virtual estiver dentro de `nickname-moderation`, use `.\.venv\Scripts\python.exe` como executável.

O módulo informa colunas ausentes, rótulos e valores de origem/status inválidos, apelidos vazios, justificativas/grupos faltantes, regras de geração ausentes para dados sintéticos, duplicatas exatas e duplicatas após normalização. A normalização é apenas para comparação; o apelido original não é alterado.

## Revisão humana da base

Liste os exemplos pendentes, incluindo sugestão, justificativa atual, origem e grupo:

```powershell
python -m src.review_nicknames list
```

Opcionalmente limite a lista com `--limit 10`. Para registrar uma decisão, use o número do registro mostrado:

```powershell
python -m src.review_nicknames review --record 1 --label permitido --reason "Revisado: nome próprio sem ofensa aparente."
```

Os rótulos aceitos são `permitido`, `inadequado` e `ambiguo`. A gravação exige justificativa não vazia; `nickname`, `source`, `generation_rule` e `root_group` são preservados. Em caso de entrada inválida ou erro na validação, o CSV não é substituído. Revisar um registro não altera os demais.

Para consultar o filtro lexical local:

```powershell
python -m src.lexical_filter "ApelidoFicticio"
```

O resultado informa status, justificativa e regra acionada. Os 76 termos-base estão revisados no registro por número de linha; as variações geradas permanecem pendentes na base curada e não alteram a lista lexical original. Nenhum termo é bloqueado automaticamente pelo filtro isolado. Substituições de caracteres e separadores podem produzir candidatos; letras repetidas só geram pedido de revisão para reduzir falsos positivos.

## Serviço local de moderação para o cadastro

O serviço HTTP reutiliza `models/nickname_moderation.joblib`, as regras lexicais e o limite de confiança de 0,60 definidos em `src.nickname_model`. O modelo classifica apelidos novos por padrões aprendidos nos exemplos revisados, não apenas por consulta a uma lista. Correspondências a exemplos revisados inadequados nas três fontes também são bloqueadas por palavras/frases completas; sequências parciais dentro de palavras não acionam essa regra. Uma previsão `inadequado` abaixo do limite retorna `revisao_necessaria`; previsões do modelo com confiança a partir de 0,60 e sem outros sinais de revisão são bloqueadas. No holdout agrupado atual há 74 exemplos (9 permitidos e 65 inadequados); a decisão automática bloqueou corretamente 27 inadequados e indevidamente 2 permitidos, deixando 38 inadequados sem bloqueio. A acurácia foi 0,838 e o macro-F1 0,680; confira `automatic_block_decision` no metadata antes de usar em produção. O modelo é experimental e pode gerar falsos positivos. Ele é carregado uma vez na inicialização; se não puder ser carregado, a inicialização falha. A entrada deve ser uma string não vazia de até 50 caracteres. A resposta JSON contém `status` (`permitido`, `inadequado` ou `revisao_necessaria`) e uma `justificativa` técnica. A API não registra o apelido em logs da aplicação.

No PowerShell, a partir da raiz deste repositório, instale as dependências no ambiente virtual já usado pelo projeto:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r .\nickname-moderation\requirements.txt
```

Inicie o serviço somente na interface local:

```powershell
Set-Location .\nickname-moderation
& ..\.venv\Scripts\python.exe -m uvicorn src.moderation_api:app --host 127.0.0.1 --port 8000
```

O frontend de desenvolvimento encaminha `/moderation-api/moderate-nickname` para esse serviço usando o proxy do Vite. Inicie o frontend em outro terminal:

```powershell
Set-Location .\ByteCraft\frontend
npm ci
npm run dev -- --host 127.0.0.1
```

O cadastro também precisa do backend Spring Boot e do banco configurado conforme `ByteCraft/backend/README.md`; inicie-o em outro terminal:

```powershell
Set-Location .\ByteCraft\backend
.\mvnw.cmd spring-boot:run
```

Para testar a API com um apelido fictício, envie `{"nickname":"ApelidoFicticio"}` como JSON para `http://127.0.0.1:8000/moderate-nickname`. Para o fluxo completo, abra `http://127.0.0.1:3030`, escolha a tela de aluno, preencha o apelido e o código da turma e clique em **COMEÇAR**. Apelidos inadequados são bloqueados; resultados que exigem revisão seguem para o login, enquanto a indisponibilidade do serviço impede o cadastro.

Este serviço local não configura autenticação, TLS ou CORS para acesso remoto. Não o exponha publicamente; uma implantação real precisa de controles de acesso, configuração de rede e avaliação apropriada do modelo antes de decisões automáticas.

## Rotulagem e uso futuro

Consulte `docs/labeling_policy.md` antes de adicionar ou revisar exemplos. Casos ambíguos ficam fora do treinamento supervisionado inicial até revisão ou adoção explícita de outra estratégia. LDNOOBW é somente fonte de termos candidatos a revisão; sua presença na lista não implica bloqueio. OLID-BR é fonte complementar de análise de vocabulário, não uma coleção de apelidos. CSV agregados e anotações JSON são mantidos como fontes distintas, pois há divergências entre rótulos.

`additional_data.json` não é incorporado automaticamente por apresentar sobreposição textual com train/test. O plano de análise futura está em `docs/olid_br_candidate_plan.md`.

A função `split_by_root_group` oferece uma divisão futura reprodutível que mantém variantes relacionadas juntas; ela não foi usada para criar splits nesta etapa.

## Treinamento e avaliação experimental

O treinamento usa `pandas`, `scikit-learn` (`TfidfVectorizer` de caracteres com n-gramas de 2 a 5 e `LogisticRegression`) e `joblib`. O vetor aplica `normalize_nickname` ao apelido original, sem substituir o valor armazenado. `src.nickname_model` combina os exemplos binários revisados das fontes curada, manual e ByteCraft; duplicatas normalizadas consistentes não são contadas duas vezes e conflitos de rótulo interrompem o treino. A avaliação usa uma única divisão reprodutível `GroupShuffleSplit` por `root_group`; o holdout não participa do ajuste final. Após calcular as métricas, o pipeline final é treinado em todos os registros elegíveis e salvo junto com metadados.

Treine novamente quando a base elegível mudar e consulte a auditoria e as métricas no relatório JSON impresso no terminal:

```powershell
python -m src.nickname_model
```

O comando interrompe o treinamento se a validação falhar, houver conflito entre rótulos binários do mesmo `root_group`, dados insuficientes ou não for possível criar um holdout com ambas as classes sem sobreposição de grupos. A classe `ambiguo` é excluída por política e sua contagem é informada. Os arquivos `models/nickname_moderation.joblib` e `models/nickname_moderation.metadata.json` contêm o pipeline completo e os metadados; carregá-los não exige retreinar.

### Treinamento inicial (antes dos casos manuais)

Este resumo é histórico e descreve o primeiro artefato, treinado somente com `nickname_examples.csv`. A auditoria encontrou 138 exemplos curados: 54 `permitido`, 74 `inadequado` e 10 `ambiguo`. Os ambíguos foram excluídos, resultando em 128 exemplos binários e 66 grupos. O modelo atual usa também os casos manuais revisados descritos acima; suas métricas e contagens estão em `models/nickname_moderation.metadata.json`.

Na avaliação agrupada inicial, com semente 42, foram usados 102 exemplos/52 grupos para treino e 26 exemplos/14 grupos para holdout, sem grupos compartilhados:

| Classe | Precisão | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| `permitido` | 0,6364 | 0,7000 | 0,6667 | 10 |
| `inadequado` | 0,8000 | 0,7500 | 0,7742 | 16 |

Acurácia: **0,7308**; macro-F1: **0,7204**. Matriz de confusão (linhas reais; colunas previstas, ordem `permitido`, `inadequado`):

```text
                 previsto permitido   previsto inadequado
real permitido             7                    3
real inadequado             4                   12
```

No holdout houve **3 falsos positivos** (permitidos previstos como inadequados) e **4 falsos negativos** (inadequados previstos como permitidos). O resultado evidencia riscos de bloqueio indevido de nomes/palavras legítimas e de não detecção de conteúdo inadequado. A comparação com o filtro lexical foi inconclusiva: ele não emitiu decisão revisada para nenhum dos 26 exemplos (cobertura 0%), então não há métricas comparáveis nem base para afirmar superioridade do modelo.

O relatório completo dessa avaliação inicial está em `reports/training_readiness.md`; os metadados atuais estão em `models/nickname_moderation.metadata.json`. As métricas iniciais são exploratórias, de uma única divisão pequena e sintética, e não estimam o desempenho em apelidos reais de crianças.

## Moderador interativo

Depois de treinar e salvar o modelo:

```powershell
python -m src.predict_nickname
```

Digite apelidos em sequência; `sair` ou `exit` encerra. Uma entrada vazia solicita outro apelido. Se o artefato não existir, o programa orienta a executar o treinamento.

O programa mostra `APARENTEMENTE ADEQUADO`, `INADEQUADO` ou `REVISÃO NECESSÁRIA`, a classe prevista e probabilidades estimadas quando disponíveis. A classe `ambiguo` não é treinada: incerteza do modelo resulta em `REVISÃO NECESSÁRIA`. O limiar experimental atual é **0,60**; abaixo dele a previsão exige revisão. Correspondência lexical ambígua/pendente ou desacordo com regra lexical revisada também exige revisão. Correspondências a exemplos revisados inadequados nas três fontes são bloqueadas por palavras/frases completas. No holdout atual, o modelo sozinho teve 4 falsos positivos e 26 falsos negativos sob a regra de bloqueio; consulte o metadata para métricas. As probabilidades não são calibradas nem garantias.

## Diagnóstico de casos manuais

`data/curated/manual_evaluation.csv` contém casos de sondagem com referências revisadas, justificativa em `notes` e `root_group` explícito. Como os rótulos foram confirmados e agora entram no treinamento e no bloqueio lexical, esses casos não podem mais ser tratados como avaliação independente.

Gere o relatório sem treinar:

```powershell
python -m src.evaluate_manual_cases
```

O comando carrega o pipeline já salvo e escreve previsões, probabilidades, correspondências e estado lexical, resultado final e motivo de revisão em `reports/manual_evaluation_diagnostic.csv`. Registra também se um apelido já consta na base elegível e se pertence ao treino ou ao holdout. Os resultados são diagnósticos dos casos conhecidos, não métricas independentes.

## Limitações

A coleção é sintética e pequena; não representa a diversidade de apelidos usados por crianças, dialetos, contexto social ou formas emergentes de assédio. A avaliação por grupos reduz vazamento de variantes, mas um holdout de uma única divisão não mede desempenho de produção nem calibra probabilidades. Como os casos manuais foram selecionados para investigar erros do modelo e agora entram no treino, as métricas dessa divisão não são avaliação independente desses casos. Métricas em dados sintéticos não demonstram segurança. Antes de qualquer uso real, são necessários dados representativos coletados com base legal e privacidade apropriadas, revisão humana consistente, validação independente, análise de falsos positivos/negativos e monitoramento. A integração local permanece experimental e não deve ser tratada como pronta para produção.

LDNOOBW continua sendo apenas uma fonte de candidatos lexicais consultivos, não uma política pronta. Comentários e rótulos do OLID-BR não são usados como apelidos rotulados. Permanecem pendências de licença/procedência do LDNOOBW e da correspondência exata dos arquivos locais do OLID-BR com a distribuição oficial.
