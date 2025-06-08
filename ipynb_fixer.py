import nbformat

path = "Chatbot.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

# Just remove the widget metadata (not outputs)
if "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]

with open(path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
