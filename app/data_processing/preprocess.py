import re


def extract_numbers(text):
    # Pattern to find the desired pattern
    pattern = re.compile(r"HISTORIA:\s*(\d+)")

    # Find all occurrences of the pattern in the text
    matches = pattern.findall(text)

    return matches


def extraer_profesional_y_texto(texto):
    # Patrón para encontrar el profesional
    patron_profesional = re.compile(r"Profesional:\s*(.*?)\n", re.DOTALL)

    # Patrón para encontrar el texto restante
    patron_texto = re.compile(r"Profesional:.*?\n(.*?)$", re.DOTALL)

    # Buscar el profesional en el texto
    match_profesional = patron_profesional.search(texto)
    profesional = (
        match_profesional.group(1).strip() if match_profesional else None
    )

    # Buscar el texto restante en el texto
    match_texto = patron_texto.search(texto)
    texto_restante = match_texto.group(1).strip() if match_texto else None

    from unstructured.cleaners.core import (
        clean_extra_whitespace,
        group_broken_paragraphs,
    )

    texto_restante = group_broken_paragraphs(texto_restante)
    texto_restante = clean_extra_whitespace(texto_restante)

    return profesional, texto_restante


def crear_documentos(texto):
    # Patrón para encontrar cada sección con su texto, la evolucion y el texto entre secciones
    patron_seccion = re.compile(
        r"\n\n(\d{1,2}/\d{1,2}/\d{4})\s+(.*?)(?=\d{1,2}/\d{1,2}/\d{4}|$)",
        re.DOTALL,
    )

    # Buscar todas las secciones en el texto
    secciones_encontradas = patron_seccion.findall(texto)

    from langchain_core.documents import Document
    from unstructured.cleaners.extract import extract_text_before

    numero_historia = extract_numbers(texto)[0]

    docs = []

    # Iterar sobre las secciones encontradas e imprimir cada una con su texto correspondiente y el profesional
    for fecha, texto_restante in secciones_encontradas:
        try:
            tipo_evolucion = extract_text_before(
                texto_restante, "Profesional:"
            )
            tipo_evolucion = tipo_evolucion.replace("\n\n", " ")
            if tipo_evolucion == "":
                tipo_evolucion = "Sin especificar"
        except:
            tipo_evolucion = "Sin especificar"

        profesional, texto_evolucion = extraer_profesional_y_texto(
            texto_restante
        )

        docs.append(
            Document(
                page_content=texto_evolucion,
                metadata={
                    "fecha": fecha,
                    "profesional": profesional,
                    "tipo_evolucion": tipo_evolucion,
                    "numero_historia": numero_historia,
                },
            )
        )

    return docs
