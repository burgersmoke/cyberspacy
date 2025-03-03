import spacy
from spacy import Language

import cyberspacy.url.url_parser
import cyberspacy.url.malicious_url_classifier
from cyberspacy._extensions import set_extensions

class PipelineFactory(object):
    def __init__(self):
        self.pipeline_names = ['malicious_url_classifier']

        set_extensions()

    def create_url_parser_pipeline(self):
        #start with blank
        nlp = spacy.blank("en")

        # break URLs into parts
        nlp.add_pipe("cyberspacy_url_parser")

        return nlp

    def create_malicious_url_classifier_pipeline(self):
        '''
        Creates a simple pipeline which processes URLs it encounters and applies
        a predictive model to determine if the URL might be malicious.
        The training of this model was performed with two datasets from Kaggle
        '''
        #start with blank
        nlp = spacy.blank("en")

        # break URLs into parts
        nlp.add_pipe("cyberspacy_url_parser")

        nlp.add_pipe("cyberspacy_malicious_url_classifier")

        return nlp
