from flask import Flask, request
import requests
import re

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
import tensorflow as tf

from bs4 import BeautifulSoup
import pickle

app = Flask(__name__)

stopWords = {" ", "a", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", 
             "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", 
             "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", 
             "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", 
             "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", 
             "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", 
             "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", 
             "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", 
             "ax", "ay", "az", "b", "ba", "back", "bc", "bd", "be", "became", "because", "become", 
             "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", 
             "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", 
             "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", 
             "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", 
             "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", 
             "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", 
             "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", 
             "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", 
             "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", 
             "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", 
             "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "ea", "each", "ec", "ed", "edu", "ee", 
             "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", 
             "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", 
             "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", 
             "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", 
             "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", 
             "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", 
             "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", 
             "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", 
             "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", 
             "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", 
             "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", 
             "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", 
             "hr", "hs", "http", "hu", "hundred", "hy", "i", "ia", "ib", "ibid", "ic", "id",
             "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", 
             "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", 
             "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", 
             "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", 
             "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", 
             "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", 
             "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", 
             "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", 
             "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", 
             "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", 
             "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", 
             "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", 
             "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", 
             "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", 
             "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", 
             "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", 
             "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", 
             "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", 
             "owing", "own", "ox", "oz", "p", "page", "pagecount", "pages", "par", "part", "particular", 
             "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", 
             "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", 
             "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", 
             "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", 
             "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", 
             "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", 
             "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", 
             "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", 
             "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", 
             "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", 
             "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", 
             "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", 
             "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", 
             "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", 
             "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", 
             "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", 
             "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", 
             "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs",
             "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", 
             "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", 
             "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", 
             "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", 
             "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", 
             "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", 
             "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", 
             "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", 
             "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", 
             "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", 
             "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", 
             "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", 
             "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", 
             "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", 
             "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", 
             "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", 
             "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", 
             "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", 
             "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"}

with open('src/app/vectorizer.pkl', 'rb') as f:
    vectorizerpkl = pickle.load(f)

with open('src/app/tsvd.pkl', 'rb') as f:
    tsvdpkl = pickle.load(f)

with open('src/app/svc.pkl', 'rb') as f:
    svcpkl = pickle.load(f)

vectorizer = TfidfVectorizer()
vectorizer.__dict__ = vectorizerpkl.__dict__

tsvd = TruncatedSVD(n_components=5)
tsvd.__dict__ = tsvdpkl.__dict__

svc = svm.SVC(kernel="poly", C=0.5, degree=4)
svc.__dict__ = svcpkl.__dict__

rns = tf.keras.models.load_model('src/app/rns.h5')
rnc = tf.keras.models.load_model('src/app/rnc.h5')
rnt = tf.keras.models.load_model('src/app/rnt.h5')

def filtro(content):
    ps = PorterStemmer()
    contenido = re.sub('[^a-zA-Z]',' ',content)
    contenido = contenido.lower()
    contenido = contenido.split()
    contenido = [ps.stem(word) for word in contenido if word not in stopWords]
    contenido = list(map(ps.stem, contenido))
    contenido = ' '.join(contenido)
    return contenido

def getInfo(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    titulo1 = soup.find('h1')
    titulo2 = soup.find('title')
    titulo = ""
    if titulo1 != None:
        titulo += titulo1.get_text()
    elif titulo2 != None:
        titulo += titulo2.get_text()
    else: 
        titulo = ""

    author1 = soup.find('span', class_='author')
    author2 = soup.find('span', class_='byline__name')
    author3 = soup.find('div', class_='author')
    author4 = soup.find('div', class_='byline__name')
    author_tag = soup.find('meta', attrs={'name': 'author'})
    author5 = author_tag['content'] if author_tag else None

    author = ""
    if author1 != None:
        author += author1.get_text()
    elif author2 != None:
        author += author2.get_text()
    elif author3 != None:
        author += author3.get_text()
    elif author4 != None:
        author += author4.get_text()
    elif author5 != None:
        author += author5    
    else: 
        author = ""    
        
    return titulo + author

def binarizador(p, pAux, x):
    for i in p:
        if i >= x:
            pAux.append(1)
        else: 
            pAux.append(0)

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/process_data', methods=['POST'])
def process_data():
    
    url = request.form['url']
    model1 = request.form['model1']
    model2 = request.form['model2']
    model3 = request.form['model3']
    model4 = request.form['model4']

    noticiaExterna = getInfo(url)
    noticiaExterna = filtro(noticiaExterna)
    noticiaExterna = [noticiaExterna]

    noticiaExternaVec = vectorizer.transform(noticiaExterna)
    noticiaExternaVec = tsvd.transform(noticiaExternaVec)

    if model1 == "true":
        predictionExterna1 = svc.predict(noticiaExternaVec)
        if(predictionExterna1 == [1]):
            return "1"
        if(predictionExterna1 == [0]):
            return "0"
        else:
            return "x"
    if model2 == "true":
        predictionExterna2 = rns.predict(noticiaExternaVec)
        predictionExterna2Aux = []
        binarizador(predictionExterna2, predictionExterna2Aux, 0.2)
        if(predictionExterna2Aux == [1]):
            return "1"
        if(predictionExterna2Aux == [0]):
            return "0"
        else:
            return "x"
    if model3 == "true":
        predictionExterna3 = rnc.predict(noticiaExternaVec)
        predictionExterna3Aux = []
        binarizador(predictionExterna3, predictionExterna3Aux, 0.45)
        if(predictionExterna3Aux == [1]):
            return "1"
        if(predictionExterna3Aux == [0]):
            return "0"
        else:
            return "x"
    if model4 == "true":
        predictionExterna4 = rnt.predict(noticiaExternaVec)
        predictionExterna4Aux = []
        binarizador(predictionExterna4, predictionExterna4Aux, 0.05)
        if(predictionExterna4Aux == [1]):
            return "1"
        if(predictionExterna4Aux == [0]):
            return "0"
        else:
            return "x"
    else:
        return "x"
    
if __name__ == '__main__':
    app.run(debug=True)