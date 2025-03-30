import nbformat

# Carrega o notebook
with open("somente_teste.ipynb", "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

# Extrai o código de todas as células
codigo = "\n".join(cell["source"] for cell in notebook.cells if cell.cell_type == "code")

# Salva em um arquivo .py
with open("somente_teste.py", "w", encoding="utf-8") as f:
    f.write(codigo)

print("Conversão concluída! 🚀")
