from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class IT_IDF:
    def __init__(self, X, y):

        self.X_train, self.X_test, self.X_dev, self.y_train, self.y_test, self.y_dev = (
            *X,
            *y,
        )

    def get_vectors(self, current_ngram, min_df, max_features):

        vectorizer = TfidfVectorizer(
            ngram_range=current_ngram,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )
        train_vectors = vectorizer.fit_transform(self.X_train)  # reference
        test_vectors = vectorizer.transform(self.X_test)
        dev_vectors = vectorizer.transform(self.X_dev)

        return vectorizer, train_vectors, test_vectors, dev_vectors

    def get_pipe(self, vectorizer, C, random_state, max_iter, solver="saga"):
        # declaring list to store stages for a pipeline
        stages = []

        stages.append(("vectorizer", vectorizer))

        # LogisticRegression
        log_reg = LogisticRegression(
            C=C, random_state=random_state, solver=solver, max_iter=max_iter
        )
        stages.append(("classifier", log_reg))

        # training
        pipe = Pipeline(stages)
        pipe.fit(self.X_train, self.y_train)
        return pipe

    ### 📊 F1-Score Summary

    # | Method | Focus | Best for... |
    # | :--- | :--- | :--- |
    # | **Micro** | Quantity | Overall accuracy (Total hits) |
    # | **Macro** | Quality | **Imbalanced data** (Treats small classes equally) |
    # | **Weighted** | Balance | Reflecting actual data distribution |

    # * **Micro**: Every sample has equal weight.
    # * **Macro**: Every class has equal weight.
    # * **Weighted**: Classes are weighted by their size.

    # def check_case_sensitivity(self):
    #     N_vectornizer_lowercaseF = (
    #         CountVectorizer(lowercase=False).fit_transform(self.X_train).shape
    #     )  # case-sensitive
    #     N_vectornizer_lowercaseT = (
    #         CountVectorizer(lowercase=True).fit_transform(self.X_train).shape
    #     )  # case-insensitive
    #     print(
    #         f"The number of case-sensitive vectorizers: {N_vectornizer_lowercaseF[1]}"
    #     )
    #     print(
    #         f"The number of case-insensitive vectorizers: {N_vectornizer_lowercaseT[1]}"
    #     )
    #     print(f"The total number of vectorizers: {N_vectornizer_lowercaseF[0]}")
