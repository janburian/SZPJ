import argparse
from nltk.stem import PorterStemmer
from datetime import datetime

from vector_model import *

if __name__ == '__main__':
    # CMD
    parser = argparse.ArgumentParser(
        prog='Natural language processing',
        description='Creates file with the 100 most relevant documents according to the queries.',
    )

    parser.add_argument('-p', '--path_input_directory', metavar='path_input_directory', type=str,
                        help='a path to the input directory')

    parser.add_argument('-q', '--path_input_query', metavar='path_input_query', type=str,
                        help='a path to the query file')

    parser.add_argument('-s', '--path_output_file', metavar='path_output_file', type=str,
                        help='a path to the output file')

    args = parser.parse_args()

    # From CMD
    path_to_documents_directory = Path(args.path_input_directory)
    path_to_queries = Path(args.path_input_query)
    path_output_file = args.path_output_file

    if path_output_file is None:
        output_filename = 'output_SZPJ_' + datetime.now().strftime("%Y%m%d_%H%M") + '.txt'
        path_output_file = os.path.join(path_to_documents_directory.parent, output_filename)


    # Debugging
    # # Input paths
    # path_to_documents_directory = Path('./documents')
    # path_to_queries = Path('query_devel.xml')
    #
    # # Output path
    # path_output_file = Path('output.txt')

    # Loading documents
    documents_dict = load_documents(path_to_documents_directory)
    queries_list = load_queries(path_to_queries)

    # Reformatting
    documents_reformatted = reformat_documents(documents_dict)
    queries_reformatted = reformat_queries(queries_list)

    # Apply stemming
    nltk.download('punkt')
    ps = PorterStemmer()

    documents_stemming = apply_stemming(documents_reformatted, ps)
    queries_stemming = apply_stemming(queries_reformatted, ps)

    # Vectorizer
    vectorizer(documents_stemming, queries_stemming, path_output_file)