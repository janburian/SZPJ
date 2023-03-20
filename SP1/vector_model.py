import os
from pathlib import Path
import xml.dom.minidom

def load_documents(path_to_directory: Path):
    res = []
    list_documents_names = os.listdir(path_to_directory)

    for document_name in list_documents_names:
        document_path = os.path.join(path_to_documents_directory, document_name)
        document = open(document_path, "r")
        res.append(document.read())
        document.close()

    return res

def load_queries(filename: str):
    res = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line_strip = line.rstrip("\n")
            res.append(line_strip)
            # list_final_phonemes.append(list_transcript)


if __name__ == '__main__':
    path_to_documents_directory = Path("./documents")
    queries_filename = "query_devel.xml"

    documents_list = load_documents(path_to_documents_directory)
    load_queries(queries_filename)
