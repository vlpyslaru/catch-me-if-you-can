import numpy
import pandas
import sklearn, sklearn.feature_extraction.text
import tldextract

from urllib.parse import urlparse
from ipaddress import ip_address
from collections import Iterable



class TimesTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, keep_years=True, keep_days=True, keep_seconds=True):
        self.keep_years = keep_years
        self.keep_days = keep_days
        self.keep_seconds = keep_seconds
    
    
    def fit(self, X, y=None):
        #sklearn.utils.check_array(X, dtype='datetime64[ns]', accept_sparse=True, force_all_finite='allow-nan', estimator='TimesTransformer')
        
        return self

    
    def transform(self, X):
        #sklearn.utils.check_array(X, dtype='datetime64[ns]', accept_sparse=True, force_all_finite='allow-nan', estimator='TimesTransformer')
        
        source_X = X
        X = []
        
        for source_times in source_X:
            times = []
            X.append(times)
            
            try:
                base_time = pandas.Timestamp(source_times[0])
            except Exception as e:
                print(source_times[0])
                raise e
            
            if self.keep_years:
                times.append(base_time.year)
                
            if self.keep_days:
                days_since_year_start = (base_time - pandas.Timestamp(year=base_time.year, month=1, day=1)).days
                times.extend([ days_since_year_start, base_time.weekday()])
                
            if not self.keep_seconds:
                continue

            seconds_since_midnight = (base_time - pandas.Timestamp(year=base_time.year, month=base_time.month, day=base_time.day)).total_seconds()
            times.append(seconds_since_midnight)
            
            delta_time = base_time
            
            for time in source_times[1:]:
                time = pandas.Timestamp(time)
                
                times.append((time - delta_time).total_seconds())
                delta_time = time
                
        return numpy.asarray(X)

    
    
class NansCountImputer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self):
        pass
    
    
    def fit(self, X, y=None):
        sklearn.utils.check_array(X, dtype=[ 'float', 'datetime64[ns]' ], accept_sparse=True, force_all_finite='allow-nan', estimator='NanCountImputer')
        
        return self

    
    def transform(self, X):
        sklearn.utils.check_array(X, dtype=[ 'float', 'datetime64[ns]' ], accept_sparse=True, force_all_finite='allow-nan', estimator='NanCountImputer')
           
        nan_counts = [ 0 ] * X.shape[0]
            
        for i in range(X.shape[0]):
            nan_counts[i] = numpy.count_nonzero(numpy.isnan(X[i]))
            
        return numpy.concatenate([ X, numpy.asarray([ nan_counts ]).T ], axis=1)



def is_ip_address(ip):
    try:
        ip_address(ip)
    except:
        return False

    return True



class DomainsExtractor:
    def __init__(self, keep_tld=True, keep_sld=True, keep_lld=True, keep_d_ngrams=True, keep_ip=True):
        self.keep_tld = keep_tld
        self.keep_sld = keep_sld
        self.keep_lld = keep_lld
        self.keep_d_ngrams = keep_d_ngrams
        self.keep_ip = keep_ip


    def extract(self, url):
        if url is None:
            return set()

        if isinstance(url, Iterable) and (type(url) != str):
            domains = set()

            for url in url:
                domains.update(self.extract(url))

            return domains

        fqdn = urlparse(url).path.lower()
        (lld, sld, tld) = tldextract.extract(fqdn)

        if (not tld) and is_ip_address(sld):
            if self.keep_ip:
                return set([ sld ])

            return set()

        domains = []
        
        if self.keep_tld and tld:
            domains.append(tld)

        if self.keep_sld and sld:
            domains.append(sld)

        if self.keep_lld and lld:
            domains.append(lld)

        if self.keep_d_ngrams:
            if self.keep_sld and self.keep_tld and sld and tld:
                domains.append('{}.{}'.format(sld, tld))

            if self.keep_lld and self.keep_sld and lld and sld:
                domains.append('{}.{}'.format(lld, sld))

            if self.keep_lld and self.keep_sld and self.keep_tld and lld and sld and tld:
                domains.append(fqdn)

        return set(domains)



class DomainsCountImputer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, count_domains=True, count_nones=True, count_ip=True):
        self.count_domains = True
        self.count_ip = True
        self.count_nones = True
        pass


    def fit(self, X, y=None):
        sklearn.utils.check_array(X, dtype='str', accept_sparse=True, force_all_finite='allow-nan', estimator='DomainsCountImputer')

        return self

    
    def get_feautre_names(self):
        names = []

        if self.count_domains:
            names.append('domains_count')

        if self.count_ip:
            names.append('ip_count')

        if self.count_nones:
            names.append('nones_count')
            
        return names
    

    def transform(self, X):
        sklearn.utils.check_array(X, dtype='str', accept_sparse=True, force_all_finite='allow-nan', estimator='DomainsSplitter')

        domains_count_index = 0
        ip_count_index = int(self.count_domains)
        nones_count_index = ip_count_index + int(self.count_ip)
        features_count = nones_count_index + 1

        X_s = X
        X = numpy.ndarray((X_s.shape[0], features_count), dtype='int')

        for r in range(X_s.shape[0]):
            for c in range(X_s.shape[1]):
                url = X_s[r, c]

                if url is None:
                    if self.count_nones:
                        X[r, nones_count_index] = X[r, nones_count_index] + 1

                    continue

                fqdn = urlparse(url).path.lower()
                (lld, sld, tld) = tldextract.extract(fqdn)
                
                if (not tld) and is_ip_address(sld):
                    if self.count_ip:
                        X[r, ip_count_index] = X[r, ip_count_index] + 1

                    continue
                    
                X[r, domains_count_index] = X[r, domains_count_index] + 1
            
        return X
    
    
    
class DomainsVectorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, domains=None, extractor=DomainsExtractor(), use_tfidf=False):
        if not isinstance(extractor, DomainsExtractor):
            raise ValueError()

        if not (isinstance(domains, Iterable) or (domains is None)):
            raise ValueError()

        self.domains_ = domains
        self.extractor = extractor
        self.use_tfidf = use_tfidf


    def fit(self, X, y=None):
        sklearn.utils.check_array(X, dtype='str', accept_sparse=True, force_all_finite='allow-nan', estimator='DomainsVectorizer')

        if self.domains_ is None:
            self.domains_ = self.extractor.extract(X)

        analyzer = lambda v: self.extractor.extract(v)
        
        Vectorizer = sklearn.feature_extraction.text.CountVectorizer
        if self.use_tfidf:
            Vectorizer = sklearn.feature_extraction.text.TfidfVectorizer
        
        self.ll_vectorizer_ = Vectorizer(analyzer=analyzer, vocabulary=self.domains_).fit(X)

        self.fitted_ = True

        return self


    def get_feature_names(self):
        sklearn.utils.validation.check_is_fitted(self, 'fitted_')
        
        names = [ None ] * len(self.ll_vectorizer_.vocabulary_)
        
        for (name, index) in self.ll_vectorizer_.vocabulary_.items():
            names[index] = name

        return names


    def transform(self, X):
        sklearn.utils.check_array(X, dtype='str', accept_sparse=True, force_all_finite='allow-nan', estimator='DomainsVectorizer')
        sklearn.utils.validation.check_is_fitted(self, 'fitted_')

        return self.ll_vectorizer_.transform(X)



# Not used anywhere so far, sklearn doesn't support categorical Decision Trees
class DomainsSplitter(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self):
        pass


    def fit(self, X, y=None):
        sklearn.utils.check_array(X, dtype='str', accept_sparse=True, force_all_finite='allow-nan', estimator='DomainsSplitter')
        
        self.features_ = []
        
        for i in range(X.shape[1]):
            self.features_.extend([ name.format(i) for name in  [ 'domain_{}_lld', 'domain_{}_sld', 'domain_{}.tld' ]])
        
        return self
    
    
    def get_feature_names(self):
        return list(self.features_)


    def transform(self, X):
        sklearn.utils.check_array(X, dtype='str', accept_sparse=True, force_all_finite='allow-nan', estimator='DomainsSplitter')
        
        sklearn.utils.validation.check_is_fitted(self, 'features_')

        X_s = X
        X = []

        for r in range(X_s.shape[0]):
            X.append([])
            
            for c in range(X_s.shape[1]):
                if X_s[r, c] is None:
                    X[r].extend([ None ] * 3)
                    continue
                
                X[r].extend(tldextract.extract(urlparse(X_s[r, c]).path.lower()))
            
        return numpy.asarray(X)