# Web Scraping + Sentimental Analyze

Nesse projeto Python, fiz um Web Scraping com análise de sentimentos focado em noticias do mercado financeiro. Abaixo, escrevi bem rápido o que foi feito, por isso pode encontrar algum erro de português ou ver que faltou algum detalhe, também tem o fato que já fiz esse código há algum tempo, e só lembrei agora de subir aqui.

> Pedi pro ChatGPT remover os ruidos (fiz bastante comentário durante o desenvolvimento do script) e formatar o código.

**Pipeline:**
1. Coleta de dados
2. Normalização
3. Pré-processamento
4. Classificação das notícias

### 1. Coleta de dados

Primeiramente coleto as urls relacionadas ao tema desejado utilizando a **API do Bing News** com a bilbioteca `requests`, após isso acesso individualmente cada URL e coleto informações como titulo e texto da noticia utilizando a biblioteca `BeautifulSoup`.

### 2. Normalização

Não encontrei nenhum padrão especifico na identação HTML dos sites de noticias em relação a uma *class* ou *id* especifico para o corpo da noticia. Por isso optei por usar todas as tags `<p>` para capturar o texto da noticia. Isso fez com que muita coisa irrelevante viesse junto, gerando a necessidade de normalizar os dados coletados. Fiz alguns estudo em cima de diversas noticias coletadas, e por meio de regex consegui melhorar a qualidade dos textos, tornando mais fiel ao corpo da noticia original.

Figura abaixo é um texto **antes** de ser normalizado.

![image](https://github.com/user-attachments/assets/90a2ad5e-1fc8-4cf4-b2aa-0369d699bdd3)

Figura abaixo é um texto **depois** de ser normalizado.

![image](https://github.com/user-attachments/assets/dd3163e3-4b87-40c7-a3a4-0130d0951773)

### 3. Pré-processamento

Após a normalização do texto, encontrei o problema de limite de tokens que conseguia analisar, que se não me engane de cabeça eram 512 tokens. Então optei por usar algumas técnicas para limpar o texto, facilitando e aprimorando a analise sentimental, e consequentemente reduzindo o número de tokens. Usei o stopwords e lemmatizer da própria biblioteca `ntlk`, tokenizer da própria `BertTokenizer`.

### 4. Classificação de noticias

Utilizei um modelo pré-treinado que encontrei na Hugging Face.

## Resultado

Na epoca que fiz o codigo, usei o ChatGPT e minha pessoa para ler as noticias e classificar manualmente, e a assertividade da classificação tinha sido de 87.6%.

![image](https://github.com/user-attachments/assets/79a1d4b9-78da-483c-aa14-758a60dc2db0)


## Melhorias

* Percebi que a API do Bing News não traz o número máximo de noticias referente ao termo, seria interessante tentar usar o Google.
  * Obs.: Já tentei algumas APIs frees de pesquisa como NewsAPI e outras. Enfrentei o mesmo problema
* Treinar um próprio modelo, sendo necessário classificar manualmente centenas ou milhares de noticias, criando um proprio Dataset
* Melhorar a coleta de dados, reduzindo a quantidade de irrelevancia e focar em trazer apenas o corpo da noticia
