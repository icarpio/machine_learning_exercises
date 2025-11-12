# create_files.py

# Lista de archivos que quieres crear
file_names = [
    "school_dropout_pred.py",
    "home_price_pred.py",
    "customer_seg_based_purchasing_behavior.py",
    "lottery_winning_pred.py",
    "automatic_fruit_sorting.py",
    "online_product_purchase_pred.py",
    "spam_email_detection.py",
    "customer_seg_purchase_pred.py",
    "designing_ai_understands_players.py",
    "Energy_consumption_prediction.py"
]


# Contenido inicial opcional para los archivos
initial_content = """# Este archivo ha sido generado automáticamente
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