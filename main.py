import os
# Criar automaticamente a pasta 'pdf/' se ela não existir
PASTA_PDF = "pdf"
if not os.path.exists(PASTA_PDF):
    os.makedirs(PASTA_PDF)
    print(f"📁 Pasta '{PASTA_PDF}' criada para armazenar os arquivos PDF.")

import json
import docx
import faiss
from langdetect import detect
from sentence_transformers import SentenceTransformer
import argostranslate.package
import argostranslate.translate

# Inicializa o modelo de embeddings e FAISS
modelo = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
index = faiss.IndexFlatL2(384)

# Pasta onde estão os .docx
PASTA_DOCS = "livros/"
MAPEAMENTO_ARQUIVO = "mapeamento.json"

# Função para traduzir texto usando Argos Translate
def traduzir_texto(texto, idioma_origem="en", idioma_destino="pt"):
    if not texto.strip():
        return "Texto vazio ou inválido."

    try:
        # Obtém os idiomas instalados
        idiomas = argostranslate.translate.get_installed_languages()
        idioma_origem_obj = next((x for x in idiomas if x.code == idioma_origem), None)
        idioma_destino_obj = next((x for x in idiomas if x.code == idioma_destino), None)

        if not idioma_origem_obj or not idioma_destino_obj:
            return f"Erro: Não foi possível encontrar os idiomas {idioma_origem} → {idioma_destino}."

        traducao = idioma_origem_obj.get_translation(idioma_destino_obj).translate(texto)
        return traducao

    except Exception as e:
        return f"Erro ao traduzir: {e}"

# Função para processar os documentos .docx e indexar por parágrafo
def processar_docs_por_paragrafo():
    mapeamento = {}
    contador = 0

    for docx_file in os.listdir(PASTA_DOCS):
        if docx_file.endswith(".docx"):
            caminho_docx = os.path.join(PASTA_DOCS, docx_file)
            print(f"📄 Tentando processar o documento: {docx_file}")

            try:
                doc = docx.Document(caminho_docx)
                encontrou_paragrafo = False

                for i, paragrafo in enumerate(doc.paragraphs):
                    texto = paragrafo.text.strip()
                    if not texto or len(texto) < 5:
                        continue

                    encontrou_paragrafo = True
                    vetor = modelo.encode([texto])[0]
                    index.add(vetor.reshape(1, -1))

                    mapeamento[contador] = {
                        "docx": docx_file,
                        "paragrafo": i + 1,
                        "texto": texto
                    }
                    contador += 1

                if not encontrou_paragrafo:
                    print(f"⚠️ Aviso: Nenhum parágrafo válido encontrado em {docx_file}")

            except Exception as e:
                print(f"❌ Erro ao processar o documento {docx_file}: {e}")

    with open(MAPEAMENTO_ARQUIVO, 'w') as f:
        json.dump(mapeamento, f)
    print("📂 Processamento concluído e mapeamento salvo.")

# Função para buscar parágrafos relacionados e traduzir para português, se necessário
def buscar_trechos_semanticos(frase, top_k=10):
    idioma_busca = detect(frase)

    if idioma_busca == "pt":
        frase = traduzir_texto(frase, "pt", "en")

    vetor_busca = modelo.encode([frase])[0].reshape(1, -1)
    _, indices = index.search(vetor_busca, top_k)

    with open(MAPEAMENTO_ARQUIVO, 'r') as f:
        mapeamento = json.load(f)

    resultados_por_doc = {}
    for idx in indices[0]:
        if str(idx) in mapeamento:
            resultado = mapeamento[str(idx)]
            docx_file = resultado["docx"]
            texto_paragrafo = resultado["texto"]

            if detect(texto_paragrafo) != "pt":
                texto_paragrafo = traduzir_texto(texto_paragrafo, "en", "pt")

            if docx_file not in resultados_por_doc:
                resultados_por_doc[docx_file] = []

            resultados_por_doc[docx_file].append(texto_paragrafo)

    resultados_finais = {doc: parags[:3] for doc, parags in resultados_por_doc.items()}
    return resultados_finais

# Processo principal
if __name__ == "__main__":
    print("🔄 Processando documentos .docx...")
    processar_docs_por_paragrafo()

    # Define frase de busca com base no ambiente
    modo_teste = os.getenv("MODO_TESTE", "0") == "1"

    if modo_teste:
        frase_busca = "como ser criativo"
        print(f"\n💡 Frase de teste usada: {frase_busca}")
    else:
        frase_busca = input("Digite a frase em português para buscar: ")

    resultados = buscar_trechos_semanticos(frase_busca)

    print("\n📌 Resultados encontrados:")
    for doc, parags in resultados.items():
        print(f"\n📄 Documento: {doc}")
        for i, paragrafo in enumerate(parags, 1):
            print(f"\n🔹 Parágrafo {i}: {paragrafo[:500]}...")
