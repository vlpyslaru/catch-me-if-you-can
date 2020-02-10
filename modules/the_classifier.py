import sklearn, sklearn.compose, sklearn.model_selection, sklearn.impute, sklearn.ensemble, sklearn.feature_selection, sklearn.linear_model, sklearn.naive_bayes

import modules.transformers as transformers

def build_the_classifier(time_columns, domain_columns, domains):
    rf_data_transformers = sklearn.pipeline.Pipeline([
        ('times_transformer', transformers.TimesTransformer()),
        ('times_nans_count_imputer', transformers.NansCountImputer()),
        ('times_nans_zero_imputer', sklearn.impute.SimpleImputer(strategy='constant', fill_value=0)),
    ])

    rf_columns_transformer = sklearn.compose.ColumnTransformer([
        ('rf_data_transformers', rf_data_transformers, time_columns),
        ('domains_counts_imputer', transformers.DomainsCountImputer(), domain_columns),
    ])

    rf_pipeline = sklearn.pipeline.Pipeline([
        ('rf_transformers', rf_columns_transformer),
        ('rf_classifier', sklearn.ensemble.RandomForestClassifier(class_weight='balanced', random_state=42)),
    ])

    log_data_transformers = sklearn.pipeline.Pipeline([
        ('domains_vectorizer', transformers.DomainsVectorizer(domains=domains)),
        ('zero_variance_feature_selector', sklearn.feature_selection.VarianceThreshold()),
    ])

    log_columns_transformers = sklearn.compose.ColumnTransformer([
        ('logr_data_transformers', log_data_transformers, domain_columns)
    ])

    log_pipeline = sklearn.pipeline.Pipeline([
        ('log_columns_transformers', log_columns_transformers),
        ('log_classifier', sklearn.linear_model.SGDClassifier(loss='log', class_weight='balanced', random_state=42)),
    ])

    return sklearn.ensemble.StackingClassifier([
        ('log_pipeline', log_pipeline),
        ('rf_pipeline', rf_pipeline)
    ])