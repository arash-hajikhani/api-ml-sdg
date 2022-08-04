import gzip
import shutil
with gzip.open('ML_MODEL/word2vec-google-news-300.gz', 'rb') as f_in:
    with open('file.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)