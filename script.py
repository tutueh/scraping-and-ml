import requests
import json
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
import re
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification
import time

# Certifique-se de que as stop words estão disponíveis
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))
lemmatizer = WordNetLemmatizer()

# Configuração do modelo BERT e tokenizer
PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained("lucas-leme/FinBERT-PT-BR")

# Mover o modelo para GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Mapeamento de classes
class_mapping = {
    0: "Negativo",
    1: "Neutro",
    2: "Positivo"
}

# Definição da URL da API do Bing
busca_api = "https://api.bing.microsoft.com/v7.0/news/search"
key = "COLOCA SUA KEY AQUI, TEM COMO GERAR FREE"
headers_bing = {"Ocp-Apim-Subscription-Key": key}

# Função para log
def log(etapa, texto):
    horario = datetime.now(timezone.utc) - timedelta(hours=3)
    horario_formatado = horario.strftime('%d/%m/%y %H:%M')
    with open('LOG.log', 'a') as file:
        file.write(f"Data: {horario_formatado} | Etapa: {etapa} | Log: {texto}\n")

# 1. Função para classificar textos
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_class_label = class_mapping[predicted_class_index]
    confidence = probabilities[0][predicted_class_index].item()

    return predicted_class_label, confidence

# Funções de pré-processamento de texto
def remove_acento(text):
    return unidecode(text)

def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def lemmatize_text(text):
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

def remove_irrelevantes(text):
    irrelevantes = [
        'Gostaria de receber notícias', 'Assista aos melhores vídeos', 'Política de cookies', 'Veja também', 'Fonte:', 
        'Receba as reportagens', 'Siga o', 'Utilizamos cookies', 'Receba no WhatsApp', 'VÍDEOS: veja tudo sobre', 
        'Entre em grupos de', 'Digite acima e pressione Enter', 'Pressione Esc para cancelar', 'Configurações', 
        'This website uses cookies', 'Acesse nosso Grupo de WhatsApp', 'Últimas notícias', 'Assine agora', 
        'Leia mais', 'Notícias relacionadas', 'Comente esta notícia', 'Seja o primeiro a comentar', 
        'Compartilhe essa notícia', 'Assine a newsletter', 'Mais lidas', 'Publicidade', 'Últimas atualizações', 
        'Todos os direitos reservados', '©', '(c)', 'Todos os direitos', 'Informações em tempo real', 
        'Digite acima', 'Patos Hoje', 'Siga nosso canal no WhatsApp e receba em primeira mao noticias relevantes para o seu dia'
    ]
    for frase in irrelevantes:
        text = text.replace(frase, '')
    return text

def remove_caracteres_especiais(text):
    text = re.sub(r'\(?\d{2}\)?\s?\d{4,5}-?\d{4}', '', text)  # Remove telefones
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)  # Remove datas
    text = re.sub(r'\d{1,2}h\d{2}', '', text)  # Remove horas como "14h30"
    text = re.sub(r'[^A-Za-z0-9\s.,!?$]', '', text)  
    text = re.sub(r'\s+', ' ', text)  # Remove espaços extras
    return text.strip()

def preprocess_text(text):
    text = remove_acento(text)
    text = remove_irrelevantes(text)
    text = remove_caracteres_especiais(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# Adicione aqui a lista de termos de pesquisa
termos_de_pesquisa = ["cemig", "sabesp", "copel", "copasa", "Taesa", "EDP Brasil", "AES Brasil", "Alupar", "Engie Brasil", "Equatorial Energia"]

# Estrutura para armazenar o resumo geral
resumo_geral_analises = {}

headers_urls = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
    'Referer': 'https://www.google.com/',
    'Connection': 'keep-alive'
}

for termo in termos_de_pesquisa:
    print(f"Iniciando a pesquisa para o termo: {termo}")
    params = {
        "q": termo,
        "count": 100,
        "freshness": "Month",
        "sortby": "Date",
        "textFormat": "HTML",
        "mkt": "pt-BR"
    }

    resposta = requests.get(busca_api, headers=headers_bing, params=params)
    busca_resultados = resposta.json()
    urls = []

    if resposta.status_code == 200:
        log(4.1, f"Acesso com sucesso ao Bing API News para o termo: {termo}.")
        for noticia in busca_resultados['value']:
            urls.append(noticia['url'])
            log("4.1.1", f"Adicionada na lista a URL: {noticia['url']} para o termo {termo}")
    else:
        log(4.2, f"Não foi possível acessar o Bing Api News para o termo {termo}, status code: {resposta.status_code}")
        continue

    # 3. Requisição para as URLs coletadas
    news_data = {}

    for url in urls:
        response = requests.get(url, headers=headers_urls)
        if response.status_code == 200:
            log(5.1, f"Status Code: 200 | {url} para o termo {termo}")
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('h1').text.strip() if soup.find('h1') else "N/A"
            paragraphs = soup.find_all('p')
            news_text = ' '.join([paragraph.text.strip() for paragraph in paragraphs])
            news_data[url] = {'title': title, 'text': news_text, 'termo': termo}
        else:
            log("5.1.3", f"{response.status_code} | Problema a acessar a URL {url} para o termo {termo}")

    # 4. Pré-processamento e análise de sentimento
    news_analysis = []
    summary_analysis = {"Quantidade de neutros": 0, "Quantidade de positivos": 0, "Quantidade de negativos": 0}

    for i, (url, content) in enumerate(news_data.items()):
        texto_limpo = preprocess_text(content['text'])
        sentimento, score = classify_text(texto_limpo)

        news_analysis.append({
            "Termo": termo,
            "Texto": f"Notícia {i+1}",
            "Conteúdo": texto_limpo,
            "Score": round(score, 2),
            "Sentimento": sentimento
        })

        # Atualizar resumo da análise
        summary_analysis[f"Quantidade de {sentimento.lower()}s"] += 1

    # Conclusão final com base nas quantidades de sentimentos
    if summary_analysis["Quantidade de positivos"] > summary_analysis["Quantidade de negativos"]:
        conclusion = f"A análise geral das notícias para o termo {termo} é Positiva."
    elif summary_analysis["Quantidade de negativos"] > summary_analysis["Quantidade de positivos"]:
        conclusion = f"A análise geral das notícias para o termo {termo} é Negativa."
    else:
        conclusion = f"A análise geral das notícias para o termo {termo} é Neutra."

    summary_analysis["Conclusão"] = conclusion
    resumo_geral_analises[termo] = summary_analysis  # Adiciona o resumo de cada termo no dicionário geral

    # Salvar o detalhe da análise em arquivos JSON separados para cada termo
    with open(f"detalhe_analise_{termo}.json", "w", encoding="utf-8") as f:
        json.dump(news_analysis, f, ensure_ascii=False, indent=4)

    print(f"Concluída a análise para o termo: {termo}. Pausando por 10 segundos.")
    time.sleep(10)  # Pausa de 10 segundos entre cada termo

# Salvar o resumo geral em um único arquivo JSON
with open("resumo_geral_analise.json", "w", encoding="utf-8") as f:
    json.dump(resumo_geral_analises, f, ensure_ascii=False, indent=4)
