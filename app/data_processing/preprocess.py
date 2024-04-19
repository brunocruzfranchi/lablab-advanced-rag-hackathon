import re


def extract_numbers(text):
    """
    Extracts numbers from the given text using a specific pattern.

    Args:
        text (str): The text from which numbers need to be extracted.

    Returns:
        list: A list of numbers extracted from the text.

    Example:
        >>> extract_numbers("HISTORIA: 12345")
        ['12345']
    """
    # Pattern to find the desired pattern
    pattern = re.compile(r"HISTORIA:\s*(\d+)")

    # Find all occurrences of the pattern in the text
    matches = pattern.findall(text)

    return matches


def extract_professional_y_text(text):
    """
    Extracts the professional and remaining text from the given input text.

    Args:
        text (str): The input text.

    Returns:
        tuple: A tuple containing the professional and remaining text.
               The professional is a string or None if not found.
               The remaining text is a string or None if not found.
    """
    # Pattern to find the professional
    patron_professional = re.compile(r"Profesional:\s*(.*?)\n\n", re.DOTALL)

    # Pattern to find the remaining text
    patron_text = re.compile(r"Profesional:.*?\n(.*?)$", re.DOTALL)

    # Search for professional in the text
    match_professional = patron_professional.search(text)

    professional = (
        match_professional.group(1).strip() if match_professional else None
    )

    # Search for the remaining text in the text
    match_text = patron_text.search(text)
    remaining_text = match_text.group(1).strip() if match_text else None

    from unstructured.cleaners.core import (
        clean_extra_whitespace,
        group_broken_paragraphs,
    )

    remaining_text = group_broken_paragraphs(remaining_text)
    remaining_text = clean_extra_whitespace(remaining_text)

    return professional, remaining_text


def create_documents(text):
    # Pattern to find each section with its text, the text between sections
    section_pattern = re.compile(
        r"\n\n(\d{1,2}/\d{1,2}/\d{4})\s+(.*?)(?=\d{1,2}/\d{1,2}/\d{4}|$)",
        re.DOTALL,
    )

    found_sections = section_pattern.findall(text)

    from langchain_core.documents import Document
    from unstructured.cleaners.extract import extract_text_before

    num_medical_record = extract_numbers(text)[0]

    docs = []

    # Iterar sobre las secciones encontradas e imprimir cada una con su text correspondiente y el professional
    for date, remaining_text in found_sections:
        try:
            episode = extract_text_before(remaining_text, "Profesional:")
            episode = episode.replace("\n\n", " ")
            if episode == "":
                episode = "Sin especificar"
        except Exception:
            episode = "Sin especificar"

        professional, text_evolucion = extract_professional_y_text(
            remaining_text
        )

        docs.append(
            Document(
                page_content= 'Fecha: ' + date + '\n' + text_evolucion,
                metadata={
                    "date": date,
                    "professional": professional,
                    "episode": episode,
                    "num_medical_record": num_medical_record,
                },
            )
        )

    return docs
