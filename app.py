"""
Email Classifier AI - Aplicação Web com FastAPI
Modelo: DistilBERT (66M parâmetros) - Leve e Gratuito
Hospedagem: Hugging Face Spaces (Gratuito)
"""

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import io

from model import get_classifier

# Inicializar aplicação
app = FastAPI(
    title="Email Classifier AI",
    description="Classificação automática de emails usando DistilBERT",
    version="1.0.0"
)

# Configurar templates e arquivos estáticos
templates = Jinja2Templates(directory="templates")

# Inicializar modelo (lazy loading)
classifier = None

def get_model():
    global classifier
    if classifier is None:
        classifier = get_classifier()
    return classifier

@app.on_event("startup")
async def startup_event():
    """Pré-carregar modelo na inicialização"""
    print("🚀 Inicializando Email Classifier...")
    get_model()
    print("✅ Modelo DistilBERT carregado com sucesso!")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Página principal com interface web"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify")
async def classify_email(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Endpoint para classificar email via texto ou arquivo
    """
    try:
        email_text = ""

        # Processar arquivo se enviado
        if file and file.filename:
            content = await file.read()

            if file.filename.endswith('.pdf'):
                email_text = get_model().extract_text_from_pdf(content)
            elif file.filename.endswith('.docx'):
                email_text = get_model().extract_text_from_docx(content)
            elif file.filename.endswith('.txt'):
                email_text = content.decode('utf-8')
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Formato de arquivo não suportado. Use .txt, .pdf ou .docx"}
                )

        # Usar texto direto se fornecido
        elif text:
            email_text = text

        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Por favor, forneça um texto ou arquivo para classificar"}
            )

        # Classificar
        classification = get_model().classify(email_text)
        response_data = get_model().generate_response(classification, email_text)

        return JSONResponse(content={
            "success": True,
            "email_preview": email_text[:200] + "..." if len(email_text) > 200 else email_text,
            "classificacao": response_data["categoria"],
            "label": response_data["label"],
            "icone": response_data["icone"],
            "cor": response_data["cor"],
            "confianca": response_data["confianca"],
            "motivo": response_data["motivo"],
            "resposta_sugerida": response_data["resposta_sugerida"],
            "resposta_html": response_data["resposta_html"],
            "detalhes_tecnicos": response_data["detalhes_tecnicos"]
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erro ao processar: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    return {"status": "ok", "model": "distilbert-base-uncased-finetuned-sst-2-english"}

# Para Hugging Face Spaces - usar Gradio interface
import gradio as gr

def classify_gradio(email_text):
    """Função wrapper para interface Gradio"""
    if not email_text or len(email_text.strip()) < 10:
        return "⚠️ Por favor, insira um texto de email válido (mínimo 10 caracteres)."

    classification = get_model().classify(email_text)
    response_data = get_model().generate_response(classification, email_text)

    # Formatar saída
    output = f"""
## {response_data['icone']} {response_data['label']}

**Confiança:** {response_data['confianca']*100:.1f}%

**Motivo:** {response_data['motivo']}

---

### 📝 Resposta Automática Sugerida:

{response_data['resposta_html']}

---

**Detalhes Técnicos:**
- Método: {response_data['detalhes_tecnicos'].get('metodo', 'N/A')}
- Predição BERT: {response_data['detalhes_tecnicos'].get('bert_pred', 'N/A')} ({response_data['detalhes_tecnicos'].get('bert_conf', 0)*100:.1f}%)
- Predição Keywords: {response_data['detalhes_tecnicos'].get('kw_pred', 'N/A')} ({response_data['detalhes_tecnicos'].get('kw_conf', 0)*100:.1f}%)
    """
    return output

# Criar interface Gradio para Hugging Face Spaces
demo = gr.Interface(
    fn=classify_gradio,
    inputs=gr.Textbox(
        label="📧 Cole o texto do email aqui",
        placeholder="Ex: Olá, gostaria de saber o status do meu pedido #12345...",
        lines=10
    ),
    outputs=gr.Markdown(label="📊 Resultado da Classificação"),
    title="📧 Email Classifier AI",
    description="""
    **Classificação automática de emails usando Inteligência Artificial**

    Este sistema utiliza o modelo **DistilBERT** (66M parâmetros) para classificar emails em:
    - 📋 **Produtivo**: Requer ação ou resposta específica
    - ✨ **Improdutivo**: Mensagens de cortesia, agradecimentos

    *Gratuito, leve e privado - seus dados não saem deste servidor*
    """,
    examples=[
        ["Olá, gostaria de saber o status do meu pedido #12345. Está pendente há 3 dias e preciso urgente da entrega."],
        ["Prezados, agradeço muito a atenção e o excelente atendimento prestado. Feliz Natal a todos!"],
        ["Não consigo acessar minha conta. O sistema diz senha incorreta mas tenho certeza que está certa. Preciso de ajuda urgente!"],
        ["Parabéns pelo trabalho excelente no projeto! Ficou perfeito. Abraços, João"]
    ],
    theme=gr.themes.Soft()
)

# Para execução local
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
