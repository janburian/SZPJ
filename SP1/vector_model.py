import os
from pathlib import Path
import re

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
            if line == '' or line == '\n' or line == ' \n':
                continue
            else:
                line_strip = line.rstrip("\n")
                res.append(line_strip)

    return res

def reformat_queries(queries: list):
    DOC_start_tags_indices = []
    DOC_end_tags_indices = []
    idx = 0

    for element in queries:
        if element == '<DOC>':
            DOC_start_tags_indices.append(idx)
        elif element == '</DOC> ' or element == '</DOC>':
            DOC_end_tags_indices.append(idx)

        idx += 1

    separated_query_texts = separate_query_text(DOC_end_tags_indices, DOC_start_tags_indices, queries)

    return separated_query_texts


def separate_query_text(DOC_end_tags_indices, DOC_start_tags_indices, queries):
    dict_queries = {}
    for i in range(len(DOC_start_tags_indices)):
        topic = queries[DOC_start_tags_indices[i] + 1]  # element on next index after <DOC> is <DOCNO>number<\DOCNO>
        topic_identifier = int(''.join(filter(str.isdigit, topic)))
        extracted_text_start = (DOC_start_tags_indices[i] + 2)
        extracted_text_end = DOC_end_tags_indices[i]
        extracted_text_list = queries[extracted_text_start: extracted_text_end]  # text separation
        extracted_text_str = ' '.join(extracted_text_list)
        dict_queries[topic_identifier] = extracted_text_str

    return dict_queries


def reformat_documents(documents_raw):
    res = []
    for document in documents_raw:
        document_list = document.split('\n')
        document_list_pruned = []
        for element in document_list:
            element = ' '.join(element.split())  # deleting tabs (\t)
            if element in ['<html>', '</html>', '<pre>', '</pre>', ''] or 'CACM' in element or element.isdigit():
                continue
            else:
                document_list_pruned.append(element)

        document_str_pruned = ' '.join(document_list_pruned)
        res.append(document_str_pruned)

    return res


def delete_numbers():
    pass

if __name__ == '__main__':
    path_to_documents_directory = Path("./documents")
    queries_filename = "query_devel.xml"

    # Loading documents
    documents_list = load_documents(path_to_documents_directory)
    queries_list = load_queries(queries_filename)

    # Reformatting
    documents_final = reformat_documents(documents_list)
    queries_final = reformat_queries(queries_list)

    print()
