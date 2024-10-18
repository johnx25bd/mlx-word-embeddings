import collections
import pickle

with open("dataset/text8") as f:
    text8: str = f.read()


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = text.replace(".", " <PERIOD> ")
    text = text.replace(",", " <COMMA> ")
    text = text.replace('"', " <QUOTATION_MARK> ")
    text = text.replace(";", " <SEMICOLON> ")
    text = text.replace("!", " <EXCLAMATION_MARK> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace("(", " <LEFT_PAREN> ")
    text = text.replace(")", " <RIGHT_PAREN> ")
    text = text.replace("--", " <HYPHENS> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace(":", " <COLON> ")
    words = text.split()
    stats = collections.Counter(words)
    words = [word for word in words if stats[word] > 5]
    return words


corpus: list[str] = preprocess(text8)


def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    word_counts = collections.Counter(words)
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
    int_to_vocab = {ii + 1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = "<PAD>"
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab


words_to_ids, ids_to_words = create_lookup_tables(corpus)
tokens = [words_to_ids[word] for word in corpus]

dataset = (corpus, tokens, words_to_ids, ids_to_words)

with open("dataset/processed2.pkl", "wb") as f:
    pickle.dump(dataset, f)