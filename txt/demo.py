from sklearn.feature_extraction.text import TfidfVectorizer

my_data = ["Count Inversion"]

tf = TfidfVectorizer(use_idf=True)
tf.fit_transform(my_data)

idf = tf.idf_