import os
import glob
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from spacy.tokens import Token

import cyberspacy
from cyberspacy.pipelines import PipelineFactory
from cyberspacy.url.url_feature_extractor import URLFeatureExtractor

logger = logging.getLogger(__name__)

class TrainMaliciousURLClassifier(object):
    def __init__(self,
                 validation_percent = 0.20,
                 test_percent = 0.10,
                 max_instances = 50):

        self.cyberspacy_label_name = 'cyberspacy_label'
        self.validation_percent = validation_percent
        self.test_percent = test_percent
        self.max_instances = max_instances

        self.nlp = None

        factory = PipelineFactory()

        self.nlp = factory.create_url_parser_pipeline()

        # let's load the data we need
        base_dir = os.path.join(__file__, '../../../data/url')

        self.train_data_file_paths = list(glob.glob(base_dir + '/*.csv'))

        # get all CSVs...
        df_list = []
        for train_data_file_path in self.train_data_file_paths:
            logger.info(f'train_data_file_path: {train_data_file_path}')

            df = pd.read_csv(train_data_file_path)

            # let's conform the labels from these two sets for our own
            # malicious == 1 if type is not benign
            if 'type' in df.columns:
                category_mapping = {'phishing': 1, 'benign': 0, 'defacement': 1}
                df[self.cyberspacy_label_name] = df['type'].map(category_mapping)
            elif 'label' in df.columns:
                category_mapping = {'benign': 0, 'malicious': 1}
                df[self.cyberspacy_label_name] = df['label'].map(category_mapping)

            logger.info(f'len(df) for {train_data_file_path}: {len(df)}')

            df_list.append(df)

        # now let's concatenate these into one...
        self.df = pd.concat(df_list, ignore_index = True)


    def train(self):
        # let's split into train/validate/test...
        logger.info(f'Splitting into train/validate/test')

        if self.max_instances is not None:
            samples_each = int(self.max_instances / 2)
            pos_df = self.df[self.df[self.cyberspacy_label_name] == 1].head(samples_each)
            neg_df = self.df[self.df[self.cyberspacy_label_name] == 0].head(samples_each)

            self.df = pd.concat([pos_df, neg_df], ignore_index = False)

            logger.info(f'Total instances after setting a max: {len(self.df)}')

        # let's get our X (text at least) and y

        X_text = self.df['url'].tolist()

        # process them with our NLP pipeline...
        X_docs = self.nlp.pipe(X_text)

        # now lets get our URL tokens
        X_url_tokens = []
        for X_doc in X_docs:
            url_token = None
            for token in X_doc:

                if url_token is None:
                    url_token = token

                if token.like_url:
                    url_token = token

            X_url_tokens.append(url_token)

        # before we go on, let's convert these to feature dictionaries
        extractor = URLFeatureExtractor()
        X = [extractor.extract_feature_dictionary(x) for x in X_url_tokens]

        y = self.df[self.cyberspacy_label_name].tolist()

        val_test_percent = self.validation_percent + self.test_percent
        train_percent = 1.0 - val_test_percent

        X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, stratify = y,
                                                                            random_state = 77)

        test_size = int(len(X_text) * self.test_percent)

        X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest,
                                                                  test_size = test_size, stratify = y_valtest,
                                                                  random_state = 777)

        print(f'len(y_train): {len(y_train)}')
        print(f'len(y_val): {len(y_val)}')
        print(f'len(y_test): {len(y_test)}')











if __name__ == '__main__':
    classifier = TrainMaliciousURLClassifier()

    classifier.train()
