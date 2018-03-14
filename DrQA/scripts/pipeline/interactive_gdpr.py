#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive interface to full DrQA pipeline."""

#Modification of the file. Add parameter directly

import torch
import argparse
import code
import prettytable
import logging

from termcolor import colored
from drqa import pipeline
from drqa.retriever import utils

class Process():
    
    def __init__(self):
        
        self.reader_model_arg = None
        self.retriever_model_arg = None
        self.doc_db_arg = None
        self.tokenizer_arg = None
        self.candidate_file_arg = None
        self.no_cuda_arg = False
        self.gpu_arg = -1

        self.DrQA = None

    def process(self, question, candidates=None, top_n=3, n_docs=10):

        print('retriever_model_arg:',self.retriever_model_arg)
        #Start Modification 09/03/2018
        #Set a environnement variable
        import drqa.tokenizers
        drqa.tokenizers.set_default('corenlp_classpath', '/home/ubuntu/spacework/DrQA/data/corenlp/*')
        # end modification

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        if self.retriever_model_arg is None:
            self.retriever_model_arg = '/home/ubuntu/spacework/DrQA/data/gdpr/gdpr_all_en_articles-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'

        if self.doc_db_arg is None:
            self.doc_db_arg = '/home/ubuntu/spacework/DrQA/data/gdpr/gdpr_all_en_articles.db'

        # Comment the arguments
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--reader-model', type=str, default=None,
        #                     help='Path to trained Document Reader model')
        # parser.add_argument('--retriever-model', type=str, default=retriever_model,
        #                     help='Path to Document Retriever model (tfidf)')
        # parser.add_argument('--doc-db', type=str, default=doc_db,
        #                     help='Path to Document DB')
        # parser.add_argument('--tokenizer', type=str, default=None,
        #                     help=("String option specifying tokenizer type to "
        #                           "use (e.g. 'corenlp')"))
        # parser.add_argument('--candidate-file', type=str, default=None,
        #                     help=("List of candidates to restrict predictions to, "
        #                           "one candidate per line"))
        # parser.add_argument('--no-cuda', action='store_true',
        #                     help="Use CPU only")
        # parser.add_argument('--gpu', type=int, default=-1,
        #                     help="Specify GPU device id to use")
        # args = parser.parse_args()
        #end comment arguments

        # Modification 09/03/2018
        # changa the args
        cuda_arg = not self.no_cuda_arg and torch.cuda.is_available()
        if cuda_arg:
            torch.cuda.set_device(self.gpu_arg)
            logger.info('CUDA enabled (GPU %d)' % self.gpu_arg)
        else:
            logger.info('Running on CPU only.')

        if self.candidate_file_arg:
            logger.info('Loading candidates from %s' % self.candidate_file_arg)
            candidates = set()
            with open(self.candidate_file_arg) as f:
                for line in f:
                    line = utils.normalize(line.strip()).lower()
                    candidates.add(line)
            logger.info('Loaded %d candidates.' % len(candidates))
        else:
            candidates = None

        print('DrQA:',self.DrQA)
        if self.DrQA is None:
            logger.info('Initializing pipeline...')

            self.DrQA = pipeline.DrQA(
                cuda=cuda_arg,
                fixed_candidates=candidates,
                reader_model=self.reader_model_arg,
                ranker_config={'options': {'tfidf_path': self.retriever_model_arg}},
                db_config={'options': {'db_path': self.doc_db_arg}},
                tokenizer=self.tokenizer_arg
            )

        predictions = self.DrQA.process(question, candidates, top_n, n_docs, return_context=True)

        table = prettytable.PrettyTable(['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score'])

        dico_result_list = []

        for i, p in enumerate(predictions, 1):
            table.add_row([i, p['span'], p['doc_id'], '%.5g' % p['span_score'], '%.5g' % p['doc_score']])
            dico_result = {}
            dico_result['answer'] = p['span']
            dico_result['docid'] = p['doc_id']
            dico_result['docscore'] = p['span_score']
            dico_result['answerscore'] = p['doc_score']

            text = p['context']['text']
            start = p['context']['start']
            end = p['context']['end']
            output = (text[:start] +
                      colored(text[start: end], 'green', attrs=['bold']) +
                      text[end:])

            dico_result['doc'] = output

            dico_result_list.append(dico_result)

        print('Top Predictions:')
        print(table)
        print('\nContexts:')
        for p in predictions:
            text = p['context']['text']
            start = p['context']['start']
            end = p['context']['end']
            output = (text[:start] +
                      colored(text[start: end], 'green', attrs=['bold']) +
                      text[end:])
            print('[ Doc = %s ]' % p['doc_id'])
            print(output + '\n')

        for dico in dico_result_list:
            print(dico)

        return dico_result_list

# banner = """
# Interactive DrQA
# >> process(question, candidates=None, top_n=1, n_docs=5)
# >> usage()
# """


# def usage():
#     print(banner)


# code.interact(banner=banner, local=locals())