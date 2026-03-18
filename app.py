"""
Email Classifier AI - Aplicação Gradio para Hugging Face Spaces
Modelo: DistilBERT (66M parâmetros) - Leve e Gratuito
Compatible with: Gradio 6.x
"""

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import PyPDF2
import docx
import io
import re
import os
from typing import Dict, Tuple

# Inicializar modelo global (será carregado uma vez)
_classifier = None

def get_classifier():
    """Inicializa o modelo DistilBERT (lazy loading)"""
    global _classifier

    if _classifier is None:
        print("🚀 Carregando modelo DistilBERT...")
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        _classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1  # CPU
        )
        print("✅ Modelo carregado com sucesso!")

    return _classifier

# Palavras-chave para análise complementar
keywords_produtivas = [
    "solicitação", "problema", "erro", "bug", "ajuda", "suporte",
    "urgente", "reclamação", "reclamar", "status", "andamento",
    "caso", "protocolo", "atendimento", "dúvida", "dúvidas",
    "não consigo", "falha", "corrigir", "resolver", "pendente",
    "cancelar", "alterar", "atualizar", "modificar", "sistema",
    "acesso", "senha", "login", "bloqueado", "chargeback",
    "fatura", "boleto", "pagamento", "atrasado", "juros", "contestação"
]

keywords_improdutivas = [
    "feliz natal", "feliz ano novo", "obrigado", "agradeço",
    "parabéns", "bom dia", "boa tarde", "boa noite",
    "ótimo trabalho", "excelente", "maravilhoso", "perfeito",
    "satisfação", "grato", "agradecido", "abraço", "cumprimentos",
    "felicitações", "sucesso", "feliz aniversário", "ótimo"
]

def extract_text_from_pdf(file_path: str) -> str:
    """Extrai texto de PDF"""
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Erro ao ler PDF: {str(e)}"

def extract_text_from_docx(file_path: str) -> str:
    """Extrai texto de DOCX"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Erro ao ler DOCX: {str(e)}"

def extract_text_from_txt(file_path: str) -> str:
    """Extrai texto de TXT"""
    try:
        # Tentar diferentes encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                continue
        return "Erro: Não foi possível decodificar o arquivo TXT"
    except Exception as e:
        return f"Erro ao ler TXT: {str(e)}"

def preprocess_text(text: str) -> str:
    """Pré-processamento do texto"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s@.-]', ' ', text)
    return text.strip().lower()

def keyword_analysis(text: str) -> Tuple[str, float]:
    """Análise por palavras-chave"""
    text_lower = text.lower()

    score_prod = sum(1 for kw in keywords_produtivas if kw in text_lower)
    score_improd = sum(1 for kw in keywords_improdutivas if kw in text_lower)

    if score_prod > score_improd:
        return "produtivo", min(score_prod / 3, 0.9)
    elif score_improd > score_prod:
        return "improdutivo", min(score_improd / 2, 0.9)
    else:
        return "neutro", 0.5

def classify_email(email_text: str) -> Dict:
    """Classifica o email usando DistilBERT + keywords"""
    if not email_text or len(email_text.strip()) < 10:
        return {
            "categoria": "improdutivo",
            "confianca": 0.5,
            "motivo": "Texto muito curto ou vazio",
            "metodo": "fallback"
        }

    processed_text = preprocess_text(email_text)

    # Classificação BERT
    try:
        classifier = get_classifier()
        result_bert = classifier(
            processed_text[:1000],
            truncation=True,
            max_length=512
        )[0]

        if result_bert['label'] == 'POSITIVE':
            bert_category = "improdutivo"
            bert_confidence = result_bert['score']
        else:
            bert_category = "produtivo"
            bert_confidence = result_bert['score']

    except Exception as e:
        print(f"Erro BERT: {e}")
        bert_category = "neutro"
        bert_confidence = 0.0

    # Análise keywords
    kw_category, kw_confidence = keyword_analysis(processed_text)

    # Ensemble
    if bert_category == kw_category and bert_category != "neutro":
        final_category = bert_category
        final_confidence = min((bert_confidence + kw_confidence) / 2 + 0.1, 0.95)
        metodo = "ensemble_concordante"
    elif kw_category != "neutro" and bert_category == "neutro":
        final_category = kw_category
        final_confidence = kw_confidence * 0.8
        metodo = "keywords_only"
    else:
        final_category = bert_category if bert_confidence > 0.6 else kw_category
        final_confidence = bert_confidence * 0.8
        metodo = "bert_prioritario"

    if final_category == "produtivo":
        motivo = "Detectado conteúdo que requer ação: solicitação de suporte, dúvida técnica ou questão pendente."
    else:
        motivo = "Detectado conteúdo de relacionamento: agradecimento, cumprimento ou mensagem de cortesia."

    return {
        "categoria": final_category,
        "confianca": round(final_confidence, 3),
        "motivo": motivo,
        "detalhes": {
            "bert_pred": bert_category,
            "bert_conf": round(bert_confidence, 3),
            "kw_pred": kw_category,
            "kw_conf": round(kw_confidence, 3),
            "metodo": metodo
        }
    }

def generate_response(classification: Dict, email_text: str = "") -> str:
    """Gera resposta formatada em Markdown"""

    templates = {
        "produtivo": {
            "icone": "📋",
            "label": "PRODUTIVO - Requer Ação",
            "cor": "🔵",
            "saudacao": "Prezado(a) cliente,",
            "corpo": "Agradecemos seu contato. Recebemos sua solicitação e nossa equipe está analisando seu caso. Retornaremos em até 24 horas úteis com uma resposta completa.",
            "assinatura": "Atenciosamente,\nEquipe de Atendimento"
        },
        "improdutivo": {
            "icone": "✨",
            "label": "IMPRODUTIVO - Agradecimento",
            "cor": "🟢",
            "saudacao": "Olá!",
            "corpo": "Agradecemos sua mensagem! Ficamos felizes com seu contato. Caso precise de assistência técnica ou tenha alguma dúvida sobre nossos serviços, estamos à disposição.",
            "assinatura": "Um abraço,\nEquipe de Relacionamento"
        }
    }

    cat = classification["categoria"]
    template = templates[cat]

    # Personalização simples
    corpo = template["corpo"]
    if cat == "produtivo":
        if any(word in email_text.lower() for word in ["urgente", "emergência", "crítico"]):
            corpo = "Agradecemos seu contato. Identificamos a urgência em sua solicitação e nossa equipe está priorizando seu caso. Retornaremos em até 4 horas úteis."
        elif any(word in email_text.lower() for word in ["senha", "acesso", "login", "bloqueado"]):
            corpo = "Agradecemos seu contato sobre questões de acesso. Nossa equipe de suporte técnico está verificando sua situação e enviará instruções de recuperação em breve."

    resposta = f"{template['saudacao']}\n\n{corpo}\n\n{template['assinatura']}"

    output = f"""
## {template['icone']} {template['label']}

**Confiança da IA:** {classification['confianca']*100:.1f}%

**Motivo:** {classification['motivo']}

---

### 💡 Resposta Automática Sugerida:

{resposta}

---

**🔧 Detalhes Técnicos:**
- Método: {classification['detalhes']['metodo']}
- Predição BERT: {classification['detalhes']['bert_pred']} ({classification['detalhes']['bert_conf']*100:.1f}%)
- Predição Keywords: {classification['detalhes']['kw_pred']} ({classification['detalhes']['kw_conf']*100:.1f}%)
    """

    return output

def process_email(email_text: str) -> str:
    """Função principal para interface Gradio"""
    if not email_text or len(email_text.strip()) < 10:
        return "⚠️ Por favor, insira um texto de email válido (mínimo 10 caracteres)."

    try:
        classification = classify_email(email_text)
        return generate_response(classification, email_text)
    except Exception as e:
        return f"❌ Erro ao processar: {str(e)}"

def process_file(file_obj) -> str:
    """Processa arquivo enviado - CORRIGIDO para Gradio 6.x"""
    if file_obj is None:
        return "⚠️ Por favor, selecione um arquivo."

    try:
        # No Gradio 6.x, file_obj pode ser:
        # 1. Um path (str) para arquivo temporário
        # 2. Um objeto file-like com atributo .name

        file_path = None

        if isinstance(file_obj, str):
            # É um path direto
            file_path = file_obj
        elif hasattr(file_obj, 'name'):
            # É um objeto file-like
            file_path = file_obj.name
        else:
            return f"⚠️ Tipo de arquivo não reconhecido: {type(file_obj)}"

        if not file_path or not os.path.exists(file_path):
            return "⚠️ Arquivo não encontrado ou caminho inválido."

        # Detectar extensão
        file_lower = file_path.lower()

        if file_lower.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_lower.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file_lower.endswith('.txt'):
            text = extract_text_from_txt(file_path)
        else:
            return "⚠️ Formato não suportado. Use .txt, .pdf ou .docx"

        if text.startswith("Erro"):
            return f"❌ {text}"

        if not text or len(text.strip()) < 10:
            return "⚠️ Arquivo parece estar vazio ou não contém texto suficiente."

        classification = classify_email(text)
        return generate_response(classification, text)

    except Exception as e:
        import traceback
        return f"❌ Erro ao processar arquivo: {str(e)}\n\nDetalhes: {traceback.format_exc()}"

# Criar interface Gradio
demo = gr.Blocks(theme=gr.themes.Soft())

with demo:
    gr.Markdown("""
    # 📧 Email Classifier AI

    **Classificação inteligente de emails com Inteligência Artificial**

    Este sistema utiliza o modelo **DistilBERT** (66M parâmetros) para classificar emails em:
    - 📋 **Produtivo**: Requer ação ou resposta específica
    - ✨ **Improdutivo**: Mensagens de cortesia, agradecimentos

    *Gratuito, leve e privado - seus dados não saem deste servidor*
    """)

    with gr.Tabs():
        with gr.TabItem("📝 Digitar Texto"):
            input_text = gr.Textbox(
                label="Cole o texto do email aqui",
                placeholder="Ex: Olá, gostaria de saber o status do meu pedido #12345...",
                lines=10
            )
            btn_text = gr.Button("🔍 Classificar Email", variant="primary")
            output_text = gr.Markdown(label="Resultado")

            btn_text.click(fn=process_email, inputs=input_text, outputs=output_text)

        with gr.TabItem("📎 Anexar Arquivo"):
            input_file = gr.File(
                label="Selecione um arquivo (.txt, .pdf, .docx)",
                file_types=[".txt", ".pdf", ".docx"]
            )
            btn_file = gr.Button("🔍 Classificar Arquivo", variant="primary")
            output_file = gr.Markdown(label="Resultado")

            btn_file.click(fn=process_file, inputs=input_file, outputs=output_file)

    gr.Markdown("""
    ---
    ### 📊 Exemplos para testar:

    **Produtivo:** "Olá, gostaria de saber o status do meu pedido #12345. Está pendente há 3 dias."

    **Improdutivo:** "Obrigado pelo excelente atendimento! Feliz Natal a todos!"

    ---
    🤖 Powered by DistilBERT • Hospedado gratuitamente em Hugging Face Spaces
    """)

# Launch
if __name__ == "__main__":
    demo.launch()
