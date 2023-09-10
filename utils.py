import numpy as np
import rulemma
import rutokenizer
import rupostagger

def preprocess(data: list, banwords: list, speller) -> list:
    tagger = rupostagger.RuPosTagger()
    lemmatizer = rulemma.Lemmatizer()
    tokenizer = rutokenizer.Tokenizer()

    tagger.load()
    lemmatizer.load()
    tokenizer.load()

    preprocessed_data = list()
    new_data = list()
    for answer in data:
        answer = speller(answer)
        new_data.append(answer)
        tokens = tokenizer.tokenize(answer)
        tags = tagger.tag(tokens)
        lemmatized = lemmatizer.lemmatize(tags)
        lemmas = list()

        for lemma in lemmatized:
            lemmatized_word = lemma[2]
            if lemmatized_word in banwords:
                new_data.remove(answer)
                break
            lemmas.append(lemmatized_word)
        preprocessed_data.append(lemmas)
    return (preprocessed_data, new_data)


def vectorize(words, navec):
    words_vecs = [navec[word] for word in words if word in navec]
    if len(words_vecs) == 0:
        return np.zeros(300)
    words_vecs = np.array(words_vecs)
    result = words_vecs.mean(axis=0)
    return result


def load_banwords():
    with open('./banwords.txt', 'r', encoding="UTF-8") as file:
        banwords = [line.replace('\n', '') for line in file.readlines()]
    return banwords
