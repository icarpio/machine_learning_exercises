# create_files.py

# Lista de archivos que quieres crear
file_names = [
    "",
    "",
    "",
]

# Contenido inicial opcional para los archivos
initial_content = "#Archivo Generado"

"""# Este archivo ha sido generado automáticamente
# Nombre del archivo: {filename}

def main():
    print("Archivo {filename} listo para usar.")

if __name__ == "__main__":
    main()
"""

for file_name in file_names:
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(initial_content.format(filename=file_name))
    print(f"Archivo creado: {file_name}")