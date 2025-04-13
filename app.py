from flask import Flask, render_template, request
from llama_cpp import Llama

app = Flask(__name__)

llm = Llama(
    model_path="modelos/mistral-7b-instruct-v0.1.Q4_0.gguf",
    n_ctx=2048,
    verbose=False
)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    texto_original = ""

    if request.method == "POST":
        texto_original = request.form["texto"]
        prompt = f"""[INST]Reescribe el siguiente texto en español, con un tono académico, claro y profesional. 
No lo traduzcas a otro idioma, solo mejora su estilo:
"{texto_original}"[/INST]"""

        respuesta = llm(prompt=prompt, max_tokens=300, temperature=0.7, stop=["</s>"])
        resultado = respuesta["choices"][0]["text"].strip()

    return render_template("index.html", resultado=resultado, texto_original=texto_original)

if __name__ == "__main__":
    app.run(debug=True)
