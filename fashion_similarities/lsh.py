import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import itertools


class LSH:
    """
    This class implements an approximate nearest neighborhood search algorithm using the locality sensitive
    hashing framework for euclidean and cosine distance.
    :param data: a 2D numpy array consisting of the database observations in the rows and their features in the columns
    :param num_total_hashes: the total number of hash functions to be applied
    :param rows_per_band:
        in LSH, the collection of hash values for the observations is split into bands. Each band consists of
        as much hash values as specified here
    :param dist_metric: The distance metric for the LSH as well as for the exact distance of the candidates
    :param bucket_width:
        euclidean LSH relies on a hyperparameter that specifies the width of a bucket on a random plane. This can
        be specified here. If it is not, it is determined via a logic implemented in :func choose_bucket_width
    """
    @staticmethod
    def calc_dist_cosine(x, y):
        # return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)
        return 1-np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

    @staticmethod
    def calc_dist_euclidean(x, y):
        diff = x - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def product_dict(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def __init__(
            self,
            data,
            num_total_hashes,
            rows_per_band,
            hash_func="euclidean",
            dist_metric="euclidean",
            bucket_width=None,
            quantile=0.05
    ):
        self.data = data  # data in the form of observations X dimensions
        self.dim = self.data.shape[1]
        self.num_hashes = num_total_hashes
        self.num_rows = rows_per_band
        self.num_bands = int(np.ceil(num_total_hashes / rows_per_band))
        self.bucket_width = bucket_width or self.choose_bucket_width(quantile) if hash_func == "euclidean" else None
        self.hash_tables = [dict() for _table in range(self.num_bands)]
        self._generate_random_hyperplanes()
        if self.bucket_width:
            self._generate_random_offsets()
        # define the hash function
        if hash_func == "cosine":
            self.hash_func = self.get_hash_cosine
        elif hash_func == "euclidean":
            self.hash_func = self.get_hash_euclidean
        else:
            raise NotImplementedError
        # specify the function to exactly calculate the distance with
        if dist_metric == "cosine":
            self.dist_func = self.calc_dist_cosine
        elif dist_metric == "euclidean":
            self.dist_func = self.calc_dist_euclidean
        else:
            raise NotImplementedError

    def _generate_random_hyperplanes(self):
        """
        Private method. Hyperplanes need to be the same for new query points and all database point. Hence this
        function should only run within the class constructor.
        :return:
        """
        self.planes = [np.random.randn(self.dim, self.num_rows) for _band in range(self.num_bands)]

    def _generate_random_offsets(self):
        self.offset = [np.random.uniform(0, self.bucket_width, self.num_rows) for _band in range(self.num_bands)]

    def choose_bucket_width(self, quantile):
        """
        some random initialization method for bucket width if not specified
        :return:
        """
        points_ind = list(range(self.data.shape[0]))
        dists = []
        for _i in range(500):
            sample_points = np.random.choice(points_ind, 2, replace=False)
            dists.append(self.calc_dist_euclidean(*self.data[sample_points, :]))
        quant = np.quantile(dists, quantile)
        print(f"estimated {quantile * 100} % quantile distance is {quant}")
        return quant

    def get_hash_cosine(self, point, enumerator):
        """
        This function implements the hash function for cosine distance. The idea is that a random hyperplane
        classifies a point into either a positive (+1) or negative (-1) class
        :param point: and point to be hashed into a bucket
        :param enumerator: the current band / hash table to be added to
        :return:
        """
        point = np.array(point)
        return "".join(np.sign(point.dot(self.planes[enumerator])).astype("int").astype("str"))

    def get_hash_euclidean(self, point, enumerator):
        """
        This function implements the hash function for euclidean distance. The idea is that a vector is
        projected on a random direction in d dimensional space. The direction is split into buckets of width
        bucket_width. The bucket a point is projected on is its hash value. b is an offset
        :param point: and point to be hashed into a bucket
        :param enumerator: the current band / hash table to be added to
        :return:
        """
        point = np.array(point)
        return "".join(
            np.floor((point.dot(self.planes[enumerator])-self.offset[enumerator])/self.bucket_width).astype(
                "int").astype("str")
        )

    def build_hashtables(self):
        for index, input_point in enumerate(self.data):
            for i, table in enumerate(self.hash_tables):
                table.setdefault(self.hash_func(input_point, i), []).append(index)

    def get_near_duplicates(
            self,
            query: list,
            num_duplicates: int = None,
            add_query_to_db: bool = False,
            verbose=True
    ):
        query = np.array(query)
        candidates = []
        for i, table in enumerate(self.hash_tables):
            hash = self.hash_func(query, i)
            candidates.extend(table.get(hash, []))
        if add_query_to_db:
            self.extend_hash_tables(list(query))
        candidates = set(candidates)
        if not verbose:
            print(f"{len(candidates)} candidates have been retrieved. Calculate exact distance on those..")
        candidates = [(candidate, self.dist_func(query, self.data[candidate])) for candidate in candidates]
        candidates.sort(key=lambda x: x[1])
        return candidates[:num_duplicates] if num_duplicates else candidates

    def extend_hash_tables(self, add_point: list):
        self.data = np.vstack((self.data, np.array(add_point)))
        for i, table in enumerate(self.hash_tables):
            hash_value = self.hash_func(add_point, i)
            table.setdefault(hash_value, []).append(self.data.shape[0]-1)

    @staticmethod
    def get_optimal_params(data, grid, nfold=5, top_k=10, njobs=1, verbose=1):
        """
        optimizes the hyperparameters for a given lsh model
        :param grid: grid to be tuned from, e. g. {num_hashes: [100, 200], rows_per_band: [5, 10]}
        :param nfold: number of folds for cross validation
        :param top_k: top k closest items to be evaluated on
        :param njobs: parallelization option
        :return: best hyperparameter set found
        """
        def get_score(data, train_index, test_index, params, top_k):
            X_train, X_test = data[train_index], data[test_index]
            lsh = LSH(data=X_train, hash_func='euclidean', dist_metric='euclidean', **params)
            lsh.build_hashtables()
            scores = []
            for query in X_test:
                cands = lsh.get_near_duplicates(query)
                retrievals = [i[0] for i in cands[:top_k]]
                dists = {idx: LSH.calc_dist_euclidean(query, embedding) for idx, embedding in enumerate(X_train)}
                true_closest = [i[0] for i in sorted(dists.items(), key=lambda item: item[1])[:top_k]]
                # count number of true closest items that are retrieved by the lsh
                c = sum(el in true_closest for el in retrievals)
                # determine the fraction of candidates to the total population
                frac = len(cands) / len(X_train)
                scores.append(0.8 * (c / top_k) + 0.2 * (1 - frac))
            return np.mean(scores)

        parallel = Parallel(n_jobs=njobs, verbose=verbose)
        with parallel:
            kf_splits = KFold(nfold, random_state=12345)
            cart_product = LSH.product_dict(**grid)
            res_per_comb = {}
            opt_params = {}
            for i, params in enumerate(cart_product):
                print(params)
                opt_params[i] = params
                results = parallel(
                    delayed(get_score)(data, train_index, test_index, params, top_k)
                    for train_index, test_index in kf_splits.split(data)
                )
                res_per_comb[i] = np.mean(results)
            best_comb = max(res_per_comb, key=res_per_comb.get)
            max_score = max(res_per_comb.values())
        return opt_params[best_comb], max_score
