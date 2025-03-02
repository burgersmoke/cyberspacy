import os
import glob
import logging
import datetime
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import xgboost as xgb

from spacy.tokens import Token

import cyberspacy
from cyberspacy.pipelines import PipelineFactory
from cyberspacy.url.url_feature_extractor import URLFeatureExtractor

logger = logging.getLogger(__name__)

class TrainMaliciousURLClassifier(object):
    def __init__(self,
                 validation_percent = 0.20,
                 test_percent = 0.10,
                 max_instances = 500,
                 #max_instances=None,
                 cross_validate = False,
                 evaluate_test_data = False):

        self.cyberspacy_label_name = 'cyberspacy_label'
        self.validation_percent = validation_percent
        self.test_percent = test_percent
        self.max_instances = max_instances
        self.cross_validate = cross_validate
        self.evaluate_test_data = evaluate_test_data

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

        self.df = self.df.fillna(0)
        self.df = self.df.astype({self.cyberspacy_label_name: 'int32'})


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


        # let's set up a pipeline for feature encoding and a model

        n_estimators = 100
        max_depth = 8

        best_pipe = None

        pipe = Pipeline([('vectorizer', DictVectorizer()),
                         ('xgb', xgb.XGBClassifier(n_estimators = n_estimators,
                                                   max_depth = max_depth))])

        if self.cross_validate:

            n_iter = 10
            param_dist = {
                'xgb__max_depth': [2, 5, 8, 10],
                'xgb__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
            }


            logger.info('About to run RandomizedSearchCV...')

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

            search = RandomizedSearchCV(pipe,
                                        param_distributions=param_dist,
                                        cv = cv,
                                        n_iter=n_iter)
            search_result = search.fit(X_train, y_train)

            best_pipe = search

            print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))

        else:
            logger.info('About to run single fit')

            pipe.fit(X_train, y_train)

            # no CV or parameter search, so just keep this...
            best_pipe = pipe

        y_train_pred = best_pipe.predict(X_train)

        print('Train set performance:')
        print(classification_report(y_train, y_train_pred))

        y_val_pred = best_pipe.predict(X_val)

        print('Validation set performance:')
        print(classification_report(y_val, y_val_pred))


        if self.evaluate_test_data:
            print('Evaluating on test data..')


        model_output_dir = os.path.join(__file__, '../model_output/')

        now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_subdir_name = f'malicious_url_classifier_{now_str}'

        model_subdir_path = os.path.join(model_output_dir, model_subdir_name)

        logger.info(f'Prepping to write model to directory: {model_subdir_path}')

        if not os.path.exists(model_subdir_path):
            os.makedirs(model_subdir_path)

        # now we can write this out...
        model_file_name = 'malicious_url_classifier_pipeline_xgboost.pkl'
        model_file_path = os.path.join(model_subdir_path, model_file_name)

        logger.info(f'Prepping to write model to file: {model_file_path}')

        pickle.dump(best_pipe, open(model_file_path, 'wb'))

        logger.info(f'Model written to file: {model_file_path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    classifier = TrainMaliciousURLClassifier()

    classifier.train()
