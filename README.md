# Classificação de Doenças Foliares da Soja utilizando Redes Neurais Convolucionais

Este repositório contém os códigos e o conjunto de dados utilizados no desenvolvimento de um Trabalho de Conclusão de Curso (TCC) em Ciência da Computação, cujo objetivo é comparar o desempenho de redes neurais convolucionais na classificação de doenças foliares da soja.

O estudo utiliza técnicas de *transfer learning* e *fine-tuning*, avaliando três arquiteturas: **MobileNetV2, DenseNet121 e ResNet50**. A avaliação considera acurácia, precisão, *recall*, F1-score e matrizes de confusão.

## Conjunto de dados

O conjunto contém **7.203 imagens de folhas de soja**, distribuídas entre cinco classes:

- Ferrugem-asiática;
- Mancha-alvo;
- Mancha-olho-de-rã;
- Crestamento foliar de Cercospora;
- Folhas saudáveis.

> **Observação sobre a nomenclatura:** a pasta `Mancha-púrpura` do dataset corresponde à classe **Crestamento foliar de Cercospora** apresentada no artigo e nas matrizes de confusão. Após os experimentos, a denominação foi corrigida para representar a condição foliar identificada no Dryad como *Cercospora leaf blight*. Embora o crestamento foliar e a mancha-púrpura das sementes possam ser causados pelo mesmo agente, como *Cercospora kikuchii*, os termos distinguem manifestações nas folhas e nas sementes. A alteração foi apenas de nomenclatura, sem modificar as imagens, a divisão dos dados, a ordem das classes ou os resultados obtidos.

As imagens foram divididas em treinamento, validação e teste nas proporções aproximadas de **70%, 15% e 15%**, respectivamente.

| Divisão | Quantidade de imagens |
| --- | ---: |
| Treinamento | 5.043 |
| Validação | 1.081 |
| Teste | 1.079 |
| **Total** | **7.203** |

As imagens disponibilizadas estão em RGB e possuem resolução de **224 × 224 pixels**.

- [Descrição do conjunto de dados — Release v1.0](https://github.com/Miguel-Possamai-Goncalves/TCC-Classificacao-doencas-foliares-soja/releases/tag/v1.0)
- [Download do DATASET.zip](https://github.com/Miguel-Possamai-Goncalves/TCC-Classificacao-doencas-foliares-soja/releases/download/v1.0/DATASET.zip)

## Notebooks

Cada notebook realiza o treinamento e a avaliação de uma arquitetura.

| Arquitetura | Código no GitHub | Executar no Colab |
| --- | --- | --- |
| MobileNetV2 | [Notebook](MobileNetV2/MobileNetV2.ipynb) | [Abrir no Colab](https://colab.research.google.com/github/Miguel-Possamai-Goncalves/TCC/blob/main/MobileNetV2/MobileNetV2.ipynb) |
| DenseNet121 | [Notebook](DenseNet121/DenseNet121.ipynb) | [Abrir no Colab](https://colab.research.google.com/github/Miguel-Possamai-Goncalves/TCC/blob/main/DenseNet121/DenseNet121.ipynb) |
| ResNet50 | [Notebook](ResNet50/ResNet50.ipynb) | [Abrir no Colab](https://colab.research.google.com/github/Miguel-Possamai-Goncalves/TCC/blob/main/ResNet50/ResNet50.ipynb) |

## Como reproduzir os experimentos

### 1. Preparar o conjunto de dados no Google Drive

1. Baixe o arquivo `DATASET.zip` disponibilizado na Release v1.0.
2. Extraia seu conteúdo no computador.
3. A pasta extraída `DATASET` contém diretamente as pastas `Treino`, `Validação` e `Teste`. Renomeie essa pasta extraída para `Dividido_224`.
4. No Google Drive, crie a pasta `TCC` em **Meu Drive**.
5. Dentro de `TCC`, crie a pasta `DATASET`.
6. Envie a pasta `Dividido_224` para dentro de `TCC/DATASET`.

Após a montagem do Google Drive no Colab, os caminhos esperados são:

| Divisão | Caminho |
| --- | --- |
| Treinamento | `/content/drive/MyDrive/TCC/DATASET/Dividido_224/Treino` |
| Validação | `/content/drive/MyDrive/TCC/DATASET/Dividido_224/Validação` |
| Teste | `/content/drive/MyDrive/TCC/DATASET/Dividido_224/Teste` |

Cada divisão deve conter as cinco subpastas de classes, mantendo os nomes e arquivos fornecidos no ZIP.

**Correspondência dos nomes:** o diretório `Mancha-púrpura` corresponde à classe apresentada no trabalho como crestamento foliar de Cercospora. O diretório `Saudável` corresponde à classe folhas saudáveis.

**Preserve a divisão disponibilizada.** Os notebooks carregam os conjuntos existentes e não refazem a separação entre treinamento, validação e teste.

### 2. Sobre o redimensionamento

As imagens disponibilizadas no ZIP já estão em RGB e possuem resolução de **224 × 224 pixels**. Portanto, não é necessário executar novamente o script de redimensionamento para realizar os experimentos.

O script está disponível em:

[Script de redimensionamento](Script%20Redimensionamento/dataset_224x224.py)

Ele utiliza os seguintes diretórios, em relação à raiz do projeto:

| Finalidade | Diretório |
| --- | --- |
| Imagens de origem | `DATASET/Dividido` |
| Imagens redimensionadas | `DATASET/Dividido_224` |

O script preserva a organização das divisões e classes, converte as imagens para RGB, aplica redimensionamento com LANCZOS e salva os arquivos em JPEG.

### 3. Preparar o ambiente no Google Colab

1. Abra um dos notebooks pelos links apresentados neste README.
2. Salve uma cópia do notebook em seu Google Drive.
3. Selecione um ambiente de execução com GPU.
4. Execute a célula inicial que verifica o TensorFlow e lista as GPUs disponíveis.
5. Execute a célula que apresenta as versões das bibliotecas e guarde essa saída junto aos resultados.
6. Autorize a montagem do Google Drive quando solicitado.

Os notebooks utilizam:

- Python;
- TensorFlow e Keras;
- NumPy;
- scikit-learn;
- Matplotlib;
- Pillow;
- pandas.

Os notebooks publicados não fixam as versões das dependências. Confira as versões disponíveis no ambiente antes da execução e registre-as para documentar a reprodução.

É necessário acesso à internet para obter os pesos pré-treinados do ImageNet.

**Utilize uma sessão nova para cada arquitetura**, mantendo os mesmos arquivos e divisões do conjunto de dados.

### 4. Ajustes necessários antes da execução

#### MobileNetV2

Na célula que define `SEED`, `IMG_SIZE`, `BATCH_SIZE` e `NUM_CLASSES`, substitua apenas o trecho de definição dos caminhos pelo código abaixo:

```python
TRAIN_DIR = DATASET / "Treino"
TEST_DIR = DATASET / "Teste"

pastas_validacao = [
    pasta for pasta in DATASET.iterdir()
    if pasta.is_dir() and pasta.name.startswith("Valid")
]

if len(pastas_validacao) != 1:
    raise ValueError(
        "Deve existir uma única pasta de validação no dataset."
    )

VAL_DIR = pastas_validacao[0]

for pasta in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta}")
```

Esse ajuste corrige a referência inicial a `Treinamento`, pois o nome da pasta disponibilizada é `Treino`. Também identifica a pasta de validação sem depender da representação dos caracteres acentuados.

#### DenseNet121

Na primeira célula, remova o espaço antes de `import tensorflow as tf`. A linha deve começar na primeira coluna:

```python
import tensorflow as tf
```

Esses ajustes corrigem a preparação da execução e preservam os hiperparâmetros definidos nos notebooks.

### 5. Executar treinamento e avaliação

Execute as células na ordem apresentada, desde o início.

O diretório padrão do projeto é definido por:

```python
TCC = Path("/content/drive/MyDrive/TCC")
```

Se utilizar outro diretório, ajuste essa variável antes de prosseguir.

Confira as contagens de imagens e a ordem das classes antes de iniciar o treinamento. Os notebooks copiam o conjunto de dados para o armazenamento temporário do Colab:

```text
/content/DATASET/Dividido_224
```

Essa cópia é reutilizada quando já existe. Caso substitua o conjunto de dados, utilize uma sessão nova para evitar o uso de uma cópia anterior.

O fluxo de execução de cada arquitetura compreende:

1. Carregamento das imagens e aplicação do `preprocess_input` específico da arquitetura.
2. Carregamento da base pré-treinada no ImageNet.
3. Treinamento inicial do classificador, mantendo a base congelada.
4. Liberação das camadas selecionadas e execução do *fine-tuning*.
5. Carregamento do checkpoint de menor perda de validação da etapa de *fine-tuning*.
6. Avaliação no conjunto de teste.
7. Geração das métricas, dos gráficos e da matriz de confusão.

**Não execute apenas as células finais em uma sessão vazia.** Elas dependem das variáveis, dos históricos e dos modelos produzidos nas etapas anteriores.

O repositório contém os resultados exportados, mas não contém os checkpoints `.keras`. Para gerar esses arquivos, é necessário executar o treinamento.

### 6. Configurações dos experimentos

| Configuração | Valor |
| --- | --- |
| Semente em Python, NumPy e TensorFlow | 42 |
| Dimensões de entrada | 224 × 224 × 3 |
| Tamanho do lote | 32 |
| Número de classes | 5 |
| Pesos iniciais da base | ImageNet |
| Classificador | GlobalAveragePooling2D, Dropout de 0,3 e Dense com softmax |
| Regularização do kernel da camada Dense | L2 de 0,0001 |
| Otimizador | Adam |
| Taxa de aprendizagem inicial — primeira etapa | 0,001 |
| Taxa de aprendizagem inicial — *fine-tuning* | 0,000005 |
| Número máximo de épocas | 16 por etapa |
| Função de perda | `categorical_crossentropy` |
| Seleção do checkpoint | Menor `val_loss` em cada etapa |
| EarlyStopping | Paciência de 4 épocas e restauração dos melhores pesos |
| ReduceLROnPlateau | Fator de 0,2, paciência de 2 épocas e taxa mínima de 0,0000001 |
| Aumento de dados | Não utilizado |

Durante o *fine-tuning*, são liberadas as seguintes camadas, mantendo as camadas BatchNormalization congeladas:

| Arquitetura | Camadas selecionadas |
| --- | --- |
| MobileNetV2 | Camadas com prefixos `block_14`, `block_15` e `block_16`, além de `Conv_1` |
| ResNet50 | Camadas com prefixo `conv5_` |
| DenseNet121 | Camadas com prefixo `conv5_` |

### 7. Consultar os arquivos gerados

Os resultados são gravados no Google Drive, conforme a arquitetura executada:

| Arquitetura | Diretório de saída |
| --- | --- |
| MobileNetV2 | `TCC/RESULTADOS/MobileNetV2` |
| ResNet50 | `TCC/RESULTADOS/ResNet50` |
| DenseNet121 | `TCC/RESULTADOS/DenseNet121` |

As saídas incluem:

- `melhor_modelo_fase1.keras`: checkpoint selecionado na primeira etapa.
- `melhor_modelo_finetuning.keras`: checkpoint selecionado no *fine-tuning*.
- Arquivo JSON com métricas e históricos das duas etapas.
- Relatório de classificação por classe em CSV.
- Gráficos de acurácia e perda durante o treinamento.
- Imagem da matriz de confusão.

**Uma nova execução utiliza os mesmos nomes de arquivos.** Para preservar resultados anteriores, defina uma pasta diferente na variável `RESULTADOS` antes de iniciar o treinamento.

## Resultados de referência

Os arquivos publicados nas pastas `RESULTADOS` registram as seguintes acurácias no conjunto de teste:

| Arquitetura | Acurácia de teste |
| --- | ---: |
| MobileNetV2 | 91,47% |
| ResNet50 | 93,51% |
| DenseNet121 | 93,05% |

Esses valores correspondem à execução relatada no trabalho e servem como referência para a análise de novas execuções.

## Considerações sobre reprodutibilidade

Uma nova execução pode apresentar diferenças devido às versões das bibliotecas, ao hardware e a operações não determinísticas. A definição da semente não garante resultados numericamente idênticos.

Para documentar a reprodução, registre:

- As versões do Python e das bibliotecas utilizadas;
- O modelo de GPU;
- A versão do código ou o identificador do commit utilizado;
- Os parâmetros de treinamento;
- Os resultados gerados.

Mantenha os mesmos arquivos e divisões do conjunto de dados e execute cada arquitetura desde o início em uma sessão nova.
