# Classificação de Doenças Foliares da Soja utilizando Redes Neurais Convolucionais

Este repositório contém os códigos e o conjunto de dados utilizados no desenvolvimento de um Trabalho de Conclusão de Curso (TCC) em Ciência da Computação, cujo objetivo é comparar o desempenho de diferentes arquiteturas de redes neurais convolucionais (CNNs) na classificação de doenças foliares da soja.

O estudo utiliza técnicas de *transfer learning* e *fine-tuning*, avaliando três arquiteturas consolidadas: **MobileNetV2, ResNet50 e DenseNet121**. A comparação é realizada sob as mesmas condições experimentais, considerando métricas como acurácia, precisão, recall, F1-score e matrizes de confusão.

## Dataset

O conjunto de dados é composto por **7.203 imagens de folhas de soja**, distribuídas entre cinco classes:

- Ferrugem-asiática
- Mancha-alvo
- Mancha-olho-de-rã
- Crestamento foliar de Cercospora
- Folhas saudáveis

As imagens foram organizadas e preparadas para o treinamento e a avaliação dos modelos. O conjunto foi dividido em **70% para treinamento, 15% para validação e 15% para teste**.

A descrição do conjunto de dados e o arquivo para download estão disponíveis na [Release v1.0](../../releases/tag/v1.0).

## Modelos

O repositório contém os notebooks Jupyter (`.ipynb`) utilizados no treinamento e na avaliação das três arquiteturas:

- **MobileNetV2:** arquitetura convolucional leve, voltada à eficiência computacional.
- **ResNet50:** arquitetura que utiliza conexões residuais para facilitar o treinamento de redes profundas.
- **DenseNet121:** arquitetura que emprega conexões densas para favorecer a reutilização de características entre camadas.

Os modelos foram treinados em duas etapas, com ajuste inicial das camadas de classificação e posterior *fine-tuning*.

## Avaliação

O desempenho das arquiteturas foi analisado e comparado por meio de métricas de classificação e matrizes de confusão, permitindo comparar os resultados globais e identificar dificuldades na distinção entre as doenças avaliadas.

Os experimentos têm como finalidade investigar a aplicação de redes neurais convolucionais como ferramenta de apoio à identificação automatizada de doenças foliares da soja.
