# classificador_telegram

> Bot do telegram que consome o classificador de imagens construído durante a disciplina de Redes Neurais do Mestrado da FACOM - UFMS.

[Este](https://colab.research.google.com/drive/1uigSrhs4uh04VU3Xr-PTpDqlOO0Lvke7) é o colab com o código que realiza o treino da rede, e também informa algumas métricas como precisão, revocação, e uma matriz de confusão para melhor visualização. No colab, também existe uma melhor descrição do processo de criação do dataset.

## Configuração

Primeiro, salve o estado atual do modelo em um dicionário, na chave "model". Ex.:

```python
actual_state = {
    'model': model.state_dict(),
}

torch.save(actual_state, 'model.pth')
```

Adicione o arquivo `model.pth` na raíz desse repositório.

Crie um arquivo `.env`, contendo seu token do Telegram, seguindo como exemplo o arquivo `.env.example`.

No arquivo [`src/classifiers.py`](/src/classifier.py#L13), no atributo `labels`, defina uma lista ordenada contendo as classes do seu problema.

Ainda no arquivo [`src/classifiers.py`](/src/classifier.py#L51), no método `get_cached_model`, defina o seu modelo corretamente (certifique-se de que o classificador está definido com a mesma quantidade de classes que o seu problema) antes de carregar o estado do modelo utilizando o `load_state_dict()`.

## Execução

Para instalar as dependências, utilize o gerenciador de pacotes `poetry`, com o seguinte comando:

```sh
poetry install
```

Para executar o bot, utilize o seguinte comando:

```sh
poetry run python src/main.py
```

Caso não esteja utilizando o `poetry`, adapte para o seu gerenciador de pacotes.
