"""
Email Classifier usando DistilBERT - Modelo Local Leve (66M parâmetros)
40% menor que BERT, mantém 97% da performance. Roda em CPU gratuitamente.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import PyPDF2
import docx
import io
import re
from typing import Dict, Tuple
import json

class EmailClassifier:
    def __init__(self):
        """Inicializa o modelo DistilBERT para classificação binária"""
        # Usando modelo pré-treinado e fine-tuned para sentimento/classificação
        # DistilBERT é 60% mais rápido, 40% menor, mantém 97% da acurácia
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # Pipeline de classificação
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # CPU (gratuito, sem necessidade de GPU)
        )

        # Templates de resposta baseados na categoria
        self.response_templates = {
            "produtivo": {
                "saudacao": "Prezado(a) cliente,",
                "corpo": "Agradecemos seu contato. Recebemos sua solicitação e nossa equipe está analisando seu caso. Retornaremos em até 24 horas úteis com uma resposta completa.",
                "assinatura": "Atenciosamente,\nEquipe de Atendimento",
                "cor": "#2563eb",  # Azul
                "icone": "📋",
                "label": "PRODUTIVO - Requer Ação"
            },
            "improdutivo": {
                "saudacao": "Olá!",
                "corpo": "Agradecemos sua mensagem! Ficamos felizes com seu contato. Caso precise de assistência técnica ou tenha alguma dúvida sobre nossos serviços, estamos à disposição.",
                "assinatura": "Um abraço,\nEquipe de Relacionamento",
                "cor": "#059669",  # Verde
                "icone": "✨",
                "label": "IMPRODUTIVO - Agradecimento"
            }
        }

        # Palavras-chave para reforçar a classificação (ensemble simples)
        self.keywords_produtivas = [
            "solicitação", "problema", "erro", "bug", "ajuda", "suporte",
            "urgente", "reclamação", "reclamar", "status", "andamento",
            "caso", "protocolo", "atendimento", "dúvida", "dúvidas",
            "não consigo", "falha", "corrigir", "resolver", "pendente",
            "cancelar", "alterar", "atualizar", "modificar", "sistema",
            "acesso", "senha", "login", "bloqueado", "erro", "chargeback",
            "fatura", "boleto", "pagamento", "atrasado", "juros", "contestação"
        ]

        self.keywords_improdutivas = [
            "feliz natal", "feliz ano novo", "obrigado", "agradeço",
            "parabéns", "bom dia", "boa tarde", "boa noite",
            "ótimo trabalho", "excelente", "maravilhoso", "perfeito",
            "satisfação", "grato", "agradecido", "abraço", "cumprimentos",
            "felicitações", "sucesso", "feliz aniversário", "ótimo"
        ]

    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extrai texto de PDF"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            return f"Erro ao ler PDF: {str(e)}"

    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extrai texto de DOCX"""
        try:
            doc_file = io.BytesIO(file_content)
            doc = docx.Document(doc_file)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text.strip()
        except Exception as e:
            return f"Erro ao ler DOCX: {str(e)}"

    def preprocess_text(self, text: str) -> str:
        """Pré-processamento do texto"""
        # Remover espaços excessivos
        text = re.sub(r'\s+', ' ', text)
        # Remover caracteres especiais excessivos
        text = re.sub(r'[^\w\s@.-]', ' ', text)
        return text.strip().lower()

    def keyword_analysis(self, text: str) -> Tuple[str, float]:
        """Análise complementar por palavras-chave (ensemble)"""
        text_lower = text.lower()

        score_prod = sum(1 for kw in self.keywords_produtivas if kw in text_lower)
        score_improd = sum(1 for kw in self.keywords_improdutivas if kw in text_lower)

        # Normalizar scores
        total_keywords = len(self.keywords_produtivas) + len(self.keywords_improdutivas)

        if score_prod > score_improd:
            return "produtivo", min(score_prod / 3, 0.9)  # Cap em 0.9
        elif score_improd > score_prod:
            return "improdutivo", min(score_improd / 2, 0.9)
        else:
            return "neutro", 0.5

    def classify(self, text: str) -> Dict:
        """
        Classifica o email usando DistilBERT + análise de palavras-chave
        Ensemble simples para melhorar acurácia
        """
        if not text or len(text.strip()) < 10:
            return {
                "categoria": "improdutivo",
                "confianca": 0.5,
                "motivo": "Texto muito curto ou vazio",
                "metodo": "fallback"
            }

        # Pré-processar
        processed_text = self.preprocess_text(text)

        # 1. Classificação com DistilBERT (modelo principal)
        try:
            # Truncar para 512 tokens (limite do DistilBERT)
            result_bert = self.classifier(
                processed_text[:1000],  # Primeiros 1000 chars são suficientes
                truncation=True,
                max_length=512
            )[0]

            # Mapear resultado do BERT (POSITIVE/NEGATIVE) para nossas categorias
            # POSITIVE -> Improdutivo (agradecimentos, mensagens positivas)
            # NEGATIVE -> Produtivo (problemas, solicitações)
            if result_bert['label'] == 'POSITIVE':
                bert_category = "improdutivo"
                bert_confidence = result_bert['score']
            else:
                bert_category = "produtivo"
                bert_confidence = result_bert['score']

        except Exception as e:
            bert_category = "neutro"
            bert_confidence = 0.0

        # 2. Análise por palavras-chave (modelo complementar)
        kw_category, kw_confidence = self.keyword_analysis(processed_text)

        # 3. Ensemble: combinar resultados
        # Se ambos concordam, aumentar confiança
        # Se discordam, priorizar BERT mas reduzir confiança
        if bert_category == kw_category and bert_category != "neutro":
            final_category = bert_category
            final_confidence = min((bert_confidence + kw_confidence) / 2 + 0.1, 0.95)
            metodo = "ensemble_concordante"
        elif kw_category != "neutro" and bert_category == "neutro":
            final_category = kw_category
            final_confidence = kw_confidence * 0.8
            metodo = "keywords_only"
        else:
            # Priorizar BERT em caso de discordância
            final_category = bert_category if bert_confidence > 0.6 else kw_category
            final_confidence = bert_confidence * 0.8
            metodo = "bert_prioritario"

        # Determinar motivo da classificação
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

    def generate_response(self, classification: Dict, email_text: str = "") -> Dict:
        """Gera resposta automática baseada na classificação"""
        template = self.response_templates[classification["categoria"]]

        # Personalizar resposta baseada no conteúdo (simples)
        corpo = template["corpo"]

        # Se for produtivo e mencionar palavras específicas, personalizar
        if classification["categoria"] == "produtivo":
            if any(word in email_text.lower() for word in ["urgente", "emergência", "crítico"]):
                corpo = "Agradecemos seu contato. Identificamos a urgência em sua solicitação e nossa equipe está priorizando seu caso. Retornaremos em até 4 horas úteis."
            elif any(word in email_text.lower() for word in ["senha", "acesso", "login", "bloqueado"]):
                corpo = "Agradecemos seu contato sobre questões de acesso. Nossa equipe de suporte técnico está verificando sua situação e enviará instruções de recuperação em breve."

        resposta_completa = f"{template['saudacao']}\n\n{corpo}\n\n{template['assinatura']}"

        return {
            "categoria": classification["categoria"],
            "label": template["label"],
            "icone": template["icone"],
            "cor": template["cor"],
            "confianca": classification["confianca"],
            "motivo": classification["motivo"],
            "resposta_sugerida": resposta_completa,
            "resposta_html": resposta_completa.replace("\n", "<br>"),
            "detalhes_tecnicos": classification.get("detalhes", {})
        }

# Instância singleton para reutilização
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = EmailClassifier()
    return _classifier
