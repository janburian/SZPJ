import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_documents(path_to_directory: Path):
    res = {}
    list_documents_names = os.listdir(path_to_directory)

    for document_name in list_documents_names:
        document_path = os.path.join(path_to_documents_directory, document_name)
        document = open(document_path, "r")
        res[document_name] = document.read()
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
    res = {}
    for document_name in documents_raw:
        document = documents_raw[document_name]
        document_list = document.split('\n')
        document_list_pruned = []
        for element in document_list:
            element_splitted = ''.join(element.split())  # deleting tabs (\t)
            if is_number(element_splitted):
                continue
            if element in ['<html>', '</html>', '<pre>', '</pre>', '']:
                continue
            # if 'CACM' in element:
            #     str_no_CACM = element.replace('CACM', '')
            #     document_list_pruned.append(str_no_CACM)
            else:
                document_list_pruned.append(element)

        document_str_pruned = ' '.join(document_list_pruned)
        res[document_name] = document_str_pruned

    return res


def is_number(n: str):  # for detecting numbers such as floats, etc. from string
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except ValueError:
        return False

    return True


def vectorizer(data: dict, queries: dict, output_filename: str):
    data_list = []
    document_names = []
    for tuple_document_name_data in sorted(data.items()): # TODO: hashovani slovniku?
        document_name = tuple_document_name_data[0]
        document_data = tuple_document_name_data[1]

        document_names.append(os.path.splitext(document_name)[0])
        data_list.append(document_data)
    # data_tuple = tuple(data_list)

    tfidf = TfidfVectorizer(norm=None, use_idf=True, smooth_idf=False, sublinear_tf=True)  # specifikace objektu vectorizeru
    sparse_doc_term_matrix = tfidf.fit_transform(data_list)  # samotná tvorba matice slov a dokumentů
    # dense_doc_term_matrix = sparse_doc_term_matrix.toarray()  # matice v lepsim formatu
    # index = tfidf.get_feature_names_out()

    f = open(output_filename, 'w')
    for tuple_query_identifier_data in sorted(queries.items()):
        topic_identifier = tuple_query_identifier_data[0]
        query_data = tuple_query_identifier_data[1]

        query_list = [query_data]
        q = tfidf.transform(query_list)
        sim = cosine_similarity(sparse_doc_term_matrix, q)
        write_to_output_file(f, topic_identifier, document_names, sim)
    f.close()


def write_to_output_file(f: , topic_identifier: int, document_names: list, sim: np.ndarray):
    sim_list = sim.tolist()
    sim_flat_list = [item for sublist in sim_list for item in sublist]  # list of lists to list
    similarities_descending_order = np.sort(sim_flat_list)[::-1]
    relevant_document_indices = np.argsort(sim_flat_list)[::-1]

    best_100_similarities = similarities_descending_order[0:100]
    relevant_100_document_indices = relevant_document_indices[0:100]

    for similarity, document_index in zip(best_100_similarities, relevant_100_document_indices):
        f.write(str(topic_identifier) + '\t' + document_names[document_index] + '\t' + str(similarity) + '\n')


if __name__ == '__main__':
    path_to_documents_directory = Path("./documents")
    queries_filename = "query_devel.xml"

    # Loading documents
    documents_list = load_documents(path_to_documents_directory)
    queries_list = load_queries(queries_filename)

    # Reformatting
    documents_final = reformat_documents(documents_list)
    queries_final = reformat_queries(queries_list)

    # Vectorizer
    vectorizer(documents_final, queries_final, 'output.txt')
    print()
