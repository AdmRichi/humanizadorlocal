from llama_cpp import Llama

# Cargar modelo local
llm = Llama(
    model_path="modelos/mistral-7b-instruct-v0.1.Q4_0.gguf",
    n_ctx=2048,
    verbose=True
)

# Input del usuario
texto_usuario = input("Escribe el texto que quieres humanizar (tono académico):\n> ")

# Prompt reforzado en español
prompt = f"""[INST]Reescribe el siguiente texto en español, con un tono académico, claro y profesional. 
No lo traduzcas a otro idioma, solo mejora su estilo:
"{texto_usuario}"[/INST]"""

# Generar texto
respuesta = llm(
    prompt=prompt,
    max_tokens=300,
    temperature=0.7,
    stop=["</s>"]
)

# Mostrar resultado
texto_generado = respuesta["choices"][0]["text"].strip()
print("\nTexto reescrito con tono académico:")
print(f"\"{texto_generado}\"")

