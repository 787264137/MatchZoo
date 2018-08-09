from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.random import RandomSummarizer


#  heurestic method
def Luhn(rsc_file, dst_file, count):
    language = "chinese"
    parser = PlaintextParser.from_file(rsc_file, Tokenizer(language), encoding='utf-8')
    stemmer = Stemmer(language)  # 语言容器

    summarizer = LuhnSummarizer(stemmer)  # Luhn算法
    summarizer.stop_words = get_stop_words(language)
    with open(dst_file, 'w', encoding='utf-8') as f:
        for sentence in summarizer(parser.document, count):
            f.write(str(sentence))
            f.write('\n')
            print(sentence)


# heurestic method with previous statistic research
# errors
def Edmundson(rsc_file, dst_file, count):
    language = "chinese"
    parser = PlaintextParser.from_file(rsc_file, Tokenizer(language), encoding='utf-8')
    stemmer = Stemmer(language)  # 语言容器

    summarizer = EdmundsonSummarizer(stemmer)  # Luhn算法
    summarizer.stop_words = get_stop_words(language)
    with open(dst_file, 'w', encoding='utf-8') as f:
        for sentence in summarizer(parser.document, count):
            f.write(str(sentence))
            f.write('\n')
            print(sentence)


# Latent Semantic Analysis
# 对句子进行奇异值分解，按得分高低选择句子作为摘要句。
def LSA(rsc_file, dst_file, count):
    language = "chinese"
    parser = PlaintextParser.from_file(rsc_file, Tokenizer(language), encoding='utf-8')
    stemmer = Stemmer(language)  # 语言容器

    summarizer = LsaSummarizer(stemmer)  # LSA算法
    summarizer.stop_words = get_stop_words(language)
    with open(dst_file, 'w', encoding='utf-8') as f:
        for sentence in summarizer(parser.document, count):
            f.write(str(sentence))
            f.write('\n')
            print(sentence)


# Unsupervised approach inspired by algorithms PageRank and HITS
def LexRank(rsc_file, dst_file, count):
    language = "chinese"
    parser = PlaintextParser.from_file(rsc_file, Tokenizer(language), encoding='utf-8')
    stemmer = Stemmer(language)  # 语言容器

    summarizer = LexRankSummarizer(stemmer)  # Luhn算法
    summarizer.stop_words = get_stop_words(language)
    with open(dst_file, 'w', encoding='utf-8') as f:
        for sentence in summarizer(parser.document, count):
            f.write(str(sentence))
            f.write('\n')
            print(sentence)


# Unsupervised approach, also using PageRank algorithm
def TextRank(rsc_file, dst_file, count):
    language = "chinese"
    parser = PlaintextParser.from_file(rsc_file, Tokenizer(language), encoding='utf-8')
    stemmer = Stemmer(language)  # 语言容器

    summarizer = TextRankSummarizer(stemmer)  # Luhn算法
    summarizer.stop_words = get_stop_words(language)
    with open(dst_file, 'w', encoding='utf-8') as f:
        for sentence in summarizer(parser.document, count):
            f.write(str(sentence))
            f.write('\n')
            print(sentence)


# Method that is often used as a baseline in the literature.
def SumBasic(rsc_file, dst_file, count):
    language = "chinese"
    parser = PlaintextParser.from_file(rsc_file, Tokenizer(language), encoding='utf-8')
    stemmer = Stemmer(language)  # 语言容器

    summarizer = SumBasicSummarizer(stemmer)  # LSA算法
    summarizer.stop_words = get_stop_words(language)
    with open(dst_file, 'w', encoding='utf-8') as f:
        for sentence in summarizer(parser.document, count):
            f.write(str(sentence))
            f.write('\n')
            print(sentence)


# Method that greedily adds sentences to a summary so long as it decreases the KL Divergence.
def KL(rsc_file, dst_file, count):
    language = "chinese"
    parser = PlaintextParser.from_file(rsc_file, Tokenizer(language), encoding='utf-8')
    stemmer = Stemmer(language)  # 语言容器

    summarizer = KLSummarizer(stemmer)  # LSA算法
    summarizer.stop_words = get_stop_words(language)
    with open(dst_file, 'w', encoding='utf-8') as f:
        for sentence in summarizer(parser.document, count):
            f.write(str(sentence))
            f.write('\n')
            print(sentence)


def Random(rsc_file, dst_file, count):
    language = "chinese"
    parser = PlaintextParser.from_file(rsc_file, Tokenizer(language), encoding='utf-8')
    stemmer = Stemmer(language)  # 语言容器

    summarizer = RandomSummarizer(stemmer)  # LSA算法
    summarizer.stop_words = get_stop_words(language)
    with open(dst_file, 'w', encoding='utf-8') as f:
        for sentence in summarizer(parser.document, count):
            f.write(str(sentence))
            f.write('\n')
            print(sentence)


def TextRank_All(rsc_file, dst_file, count):
    with open(rsc_file, 'r', encoding='utf-8') as fr:
        docs = fr.read().split('###')[1:]
    with open(dst_file, 'w', encoding='utf-8') as fw:
        for doc_string in docs:
            if not doc_string:
                continue
            language = "chinese"
            parser = PlaintextParser.from_string(doc_string, Tokenizer(language))
            stemmer = Stemmer(language)  # 语言容器

            summarizer = TextRankSummarizer(stemmer)  # LSA算法
            summarizer.stop_words = get_stop_words(language)
            for sentence in summarizer(parser.document, count):
                fw.write(str(sentence))
                fw.write('\n')
                print(sentence)
            print('===========================================\n')
            fw.write('===========================================')


if __name__ == "__main__":
    # LSA("data/news2.txt", 'data/SumBasic_summary.txt', 2)
    rsc_file = 'data/gongwen_body.txt'
    dst_file = 'data/gongwen_summary.txt'
    TextRank_All(rsc_file, dst_file, 4)
