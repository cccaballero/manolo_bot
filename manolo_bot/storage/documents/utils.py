import uuid


def generate_document_key(filename: str) -> str:
    """
    Generates a unique key for a document to prevent naming collisions.

    :param filename: The original filename.
    :return: A unique string key (e.g., '8a2f1b_original.pdf').
    """
    unique_id = str(uuid.uuid4())[:8]
    # Clean filename to be filesystem-friendly just in case
    clean_filename = "".join([c for c in filename if c.isalnum() or c in "._-"]).strip()
    return f"{unique_id}_{clean_filename}"
