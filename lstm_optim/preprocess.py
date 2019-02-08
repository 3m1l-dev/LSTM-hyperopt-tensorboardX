"""
Run this file with python preprocess.py on the dataset to
obtain your data in the same directory as "X.dat", "y.dat",
and "embedding_matrix.dat".


This is a very minimal example of preprocessing, removing
punctuation and lowering text as our embeddings were trained
on lowercase.

We also add a few statistical features, you can add as many
as desired.

"""

from multiprocessing import Pool

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import StandardScaler

EMBED_SIZE = 300 # how big is each word vector
MAX_FEATURES = 95000 # how many unique words to use (i.e num rows in embedding vector)
MAXLEN = 70 # max number of words in a question to use

NUM_CPU = 4 # NUM_CPU for parallel processing

SEED = 2018 # SEED FOR REPRODUCEABILITY


def char_count(text, ignore_spaces=False):
    """
    Function to return total character counts in a text,
    pass the following parameter `ignore_spaces = False`
    to ignore whitespaces

    """
    if ignore_spaces:
        text = text.replace(" ", "")
    return len(text)



def clean_text(x):
    """
    Function to clean punctuation from texts.

    """

    puncts = ['|', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
              '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
              '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§',
              '″', '′', 'Â', '█', '½', 'à', '“', '★', '”', '–', '●', 'â', '►',
              '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║',
              '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†',
              '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è',
              '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（',
              '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤',
              'ï', 'Ø', '¹', '≤', '‡', '√', 'é', '&amp;', '₹', 'á', '²', 'ế',
              '청', '하', '¨', '‘', '√', '\xa0', '高', '端', '大', '气', '上',
              '档', '次', '_', '½', 'π', '#', '小', '鹿', '乱', '撞', '成', '语',
              'ë', 'à', 'ç', '@', 'ü', 'č', 'ć', 'ž', 'đ', '°', 'द', 'े', 'श',
              '्', 'र', 'ो', 'ह', 'ि', 'प', 'स', 'थ', 'त', 'न', 'व', 'ा',
              'ल', 'ं', '林', '彪', '€', '\u200b', '˚', 'ö', '~', '—', '越',
              '人', 'च', 'म', 'क', 'ु', 'य', 'ी', 'ê', 'ă', 'ễ', '∞', '抗',
              '日', '神', '剧', '，', '\uf02d', '–', 'ご', 'め', 'な', 'さ', 'い',
              'す', 'み', 'ま', 'せ', 'ん', 'ó', 'è', '£', '¡', 'ś', '≤', '¿',
              'λ', '魔', '法', '师', '）', 'ğ', 'ñ', 'ř', '그', '자', '식', '멀',
              '쩡', '다', '인', '공', '호', '흡', '데', '혀', '밀', '어', '넣', '는',
              '거', '보', '니', 'ǒ', 'ú', '️', 'ش', 'ه', 'ا', 'د', 'ة', 'ل', 'ت',
              'َ', 'ع', 'م', 'ّ', 'ق', 'ِ', 'ف', 'ي', 'ب', 'ح', 'ْ', 'ث', '³', '饭',
              '可', '以', '吃', '话', '不', '讲', '∈', 'ℝ', '爾', '汝', '文', '言',
              '∀', '禮', 'इ', 'ब', 'छ', 'ड', '़', 'ʒ', '有', '「', '寧', '錯',
              '殺', '一', '千', '絕', '放', '過', '」', '之', '勢', '㏒', '㏑',
              'ू', 'â', 'ω', 'ą', 'ō', '精', '杯', 'í', '生', '懸', '命', 'ਨ',
              'ਾ', 'ਮ', 'ੁ', '₁', '₂', 'ϵ', 'ä', 'к', 'ɾ', '\ufeff', 'ã', '©',
              '\x9d', 'ū', '™', '＝', 'ù', 'ɪ', 'ŋ', 'خ', 'ر', 'س', 'ن', 'ḵ', 'ā']

    x = x.lower()
    for punct in puncts:
        if punct in x:  # adding this line significantly speeds up performance
            x = x.replace(punct, f' {punct} ')
    return x

def parallelize_dataframe(df, func, num_cpu=NUM_CPU):
    """
    Takes an input dataframe, a function to apply and
    number of CPUs as arguments.

    """

    df_split = np.array_split(df, num_cpu)
    pool = Pool(num_cpu)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def parallel_preproc(df):
    """
    Preproc function to parallelize.

    """
    df['question_text'] = df['question_text'].apply(lambda x: str(x))
    df['q_length'] = df['question_text'].apply(lambda x: char_count(x))
    df["question_text"] = df["question_text"].apply(lambda x: x.lower())
    df["question_text"] = df["question_text"].apply(lambda x: clean_text(x))

    return df

def load_glove(word_index, embed_size=EMBED_SIZE, max_features=MAX_FEATURES):
    """
    Loading glove embeddings.

    """
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word, *arr):
        """
        Getting word coefficients from embedding file
        """
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


if __name__ == "__main__":

    np.random.seed(SEED)

    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    train_df = parallelize_dataframe(train_df, parallel_preproc)
    test_df = parallelize_dataframe(test_df, parallel_preproc)

    features = train_df.drop(['qid', 'question_text', 'target'], axis=1).fillna(0)
    val_features = test_df.drop(['qid', 'question_text'], axis=1).fillna(0)

    ## Scale statistical features.
    ss = StandardScaler()
    ss.fit(np.vstack((features, val_features)))
    features = ss.transform(features)
    val_features = ss.transform(val_features)

    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_X)+list(test_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences
    train_X = pad_sequences(train_X, maxlen=MAXLEN)
    test_X = pad_sequences(test_X, maxlen=MAXLEN)

    ## Concat with statistical features
    train_X = np.concatenate((train_X, features), axis=1)
    test_X = np.concatenate((test_X, val_features), axis=1)

    ## Get the target values
    train_y = train_df['target'].values

    #shuffling the data
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]

    word_index = tokenizer.word_index

    embedding_matrix = load_glove(word_index)

    train_X.dump("X.dat")
    train_y.dump("y.dat")

    embedding_matrix.dump("embedding_matrix.dat")

    print("Preprocessing complete.")
