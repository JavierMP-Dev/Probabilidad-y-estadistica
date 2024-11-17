import random

def generar_datos(archivo_salida, n_casos=100):
    """
    Genera datos de prueba aleatorios y los guarda en un archivo.
    :param archivo_salida: Nombre del archivo donde se guardarán los datos.
    :param n_casos: Número de casos a generar.
    """
    caracteristicas = ["sí", "no"]
    etiquetas = ["spam", "no spam"]

    with open(archivo_salida, "w") as f:
        for _ in range(n_casos):
            # Generar datos aleatorios
            x1 = random.choice(caracteristicas)  # Primera característica
            x2 = random.choice(caracteristicas)  # Segunda característica
            etiqueta = random.choice(etiquetas)  # Etiqueta

            # Guardar en el archivo
            f.write(f"{x1},{x2},{etiqueta}\n")

    print(f"Archivo {archivo_salida} generado con {n_casos} casos.")

# Generar datos y guardar en "datos.txt"
generar_datos("datos.txt", n_casos=100)
