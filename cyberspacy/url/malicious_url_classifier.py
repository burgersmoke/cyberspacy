import logging
import os
import pickle

from spacy.language import Language
from spacy.tokens import Span, Doc
from cyberspacy.url.url_feature_extractor import URLFeatureExtractor

logger = logging.getLogger(__name__)

@Language.factory("cyberspacy_malicious_url_classifier")
class MaliciousURLClassifier(object):
    def __init__(self,
                 nlp: Language,
                 name: str = "cyberspacy_malicious_url_classifier",
                 model_pipeline_relative_path: str = 'url/malicious_url_classifier/malicious_url_classifier_pipeline_xgboost.pkl'):
        self.nlp = nlp
        self.name = name
        self.model_pipeline_relative_path = model_pipeline_relative_path

        self.feature_extractor = URLFeatureExtractor()

        # let's find the path for our model, relative to this file...
        models_dir = os.path.abspath(os.path.join(__file__, '../../../models'))

        model_full_path = os.path.join(models_dir, self.model_pipeline_relative_path)

        logger.info(f'About to load malicious URL classifier from : {model_full_path}')

        self.pipe = pickle.load(open(model_full_path, 'rb'))

    def __call__(self, doc: Doc) -> Doc:
        # let's classify all URL entities
        for token_idx, token in enumerate(doc):
            if token.like_url:
                token._.URL_malicious_classification = False

                # let's get the features for this URL token so we can classify it...
                url_features = self.feature_extractor.extract_feature_dictionary(token)

                url_model_pred_array = self.pipe.predict(url_features)
                url_model_pred = url_model_pred_array[0]

                if url_model_pred == 1:
                    token._.URL_malicious_classification = True



        return doc