# File for deployment of the model

import gradio as gr
import joblib
import re, nltk
from nltk.corpus import stopwords
from nltk.tokenize import  RegexpTokenizer
from nltk import pos_tag
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

# Defining parameter and function for preprocessing

# Helper function for cleaning text and removing other irrelevant data
def clean_text(description):

    #lowercase the description
    description=description.lower()

    # Remove URLs that end with .com
    description = re.sub(r'\S+\.com', '', description)

    #remove the @mention
    description = re.sub("@[A-Za-z0-9_]+","", description)

    #remove the hashtags
    description = re.sub("#[A-Za-z0-9_]+","", description)

    #remove any links 
    description = re.sub(r"http\S+", "", description)
    description = re.sub(r"www.\S+", "", description)

    #remove punctuation 
    description = re.sub('[()!?]', ' ', description)
    description = re.sub('\[.*?\]',' ', description)

    #remove non alphabetic words 
    description = re.sub("[^a-z]"," ", description)

    # return description without unnecessary whitespace
    return description.strip()

# Making list for relevant  stopwords

# Stopwords and common words
stops = set(stopwords.words("english"))  
alphabets = set("abcdefghijklmnopqrstuvwxyz")

# Prepositions
prepositions = set([
    "about", "above", "across", "after", "against", "among", "around", "at",
    "before", "behind", "below", "beside", "between", "by", "down", "during",
    "for", "from", "in", "inside", "into", "near", "of", "off", "on", "out",
    "over", "through", "to", "toward", "under", "up", "with", "aboard", "along",
    "amid", "as", "beneath", "beyond", "but", "concerning", "considering",
    "despite", "except", "following", "like", "minus", "onto", "outside", "per",
    "plus", "regarding", "round", "since", "than", "till", "underneath", "unlike",
    "until", "upon", "versus", "via", "within", "without"
])

# Conjunctions
conjunctions = set([
    "and", "but", "for", "nor", "or", "so", "yet",
    "both", "either", "neither", "not", "only", "whether"
])

# Other words and additional stopwords
others = set([
    "ã", "å", "ì", "û", "ûªm", "ûó", "ûò", "ìñ", "ûªre", "ûªve", "ûª", 
    "ûªs", "ûówe", "among", "get", "onto", "shall", "thrice", 
    "thus", "twice", "unto", "us", "would", "rs"
])

# Common ecommerce words
common_ecommerce_words = set([
    "shop", "shopping", "buy", "genuine", "product", "store", "day", "replacement",
    "good", "description", "purchase", "checkout", "cart", "details", "discount",
    "offer", "deal", "sale", "item", "voucher", "coupon", "promo", "promotion",
    "buying", "selling", "seller", "buyer", "payment", "free", "order", "returns",
    "exchange", "refund", "customer", "service", "support", "review", "rating",
    "online", "offline", "delivery", "shipping", "track", "cash", "prices", 
    "transaction", "secure", "feature", "guarantee", "fast", "easy", "reliable",
    "safe", "doorstep", "discounted", "affordable", "cheap", "quality", "brand",
    "stock", "new", "latest", "trending", "hot", "exclusive"
])

# Ecommerce platform names
ecommerce_platforms = set(["flipkart", "amazon", "mintra", "snapdeal"])

# Combine all stopwords ensuring no overlap
all_stopwords = stops | alphabets | prepositions | conjunctions | others | common_ecommerce_words | ecommerce_platforms

# stopwords removal function
def remove_stops(description):
    description = ' '.join([word for word in description.split() if word not in (all_stopwords)])
    return description.strip()

# keeping relevant part of speech
regexp = RegexpTokenizer(r"[\w']+")

def keep_pos(text):
    # Tokenize and tag POS
    tokens_tagged = pos_tag(regexp.tokenize(text))
    
    # Define POS tags to keep
    keep_tags = {'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                 'JJ', 'JJR', 'JJS',        # Adjectives
                 'RB', 'RBR', 'RBS',        # Adverbs
                 'VB', 'VBD', 'VBG', 'VBN', # Verbs
                 'FW'}                      # Foreign words
    
    # Filter and join kept words
    return " ".join(word for word, tag in tokens_tagged if tag in keep_tags)


# # Stemming function
# stemmer = PorterStemmer()
# def text_stemmer(description):
#     stemmed_words = " ".join([stemmer.stem(word) for word in description.split()])
#     return stemmed_words

# lemmatization function
lemmatizer = spacy.load("en_core_web_sm", disable = ['parser', 'ner']) # using spacy lemmatization for high accuracy and speed

def text_lemmatizer(description):
    lemmatize_words = " ".join([token.lemma_ for token in lemmatizer(description)])
    return lemmatize_words


# Text normalization
def text_normalization(description):

    description = clean_text(description)
    description = text_lemmatizer(description) # using lemmatizer for high accuracy alternately use stemmer for speed.
    # description = text_stemmer(description)

    description = keep_pos(description)
    description = remove_stops(description)
    return description


# importing model and tf-idf vectorizer
tuned_SGD_model = joblib.load(r'models\tuned_SGD_model.pkl')
TfidfVec = joblib.load(r'models\tfidf_vectorizer.pkl')



# Define category mapping
category_mapping = {
    0: 'Clothing',
    1: 'Footwear',
    2: 'Pens & Stationery',
    3: 'Bags, Wallets & Belts',
    4: 'Home Decor & Festive Needs',
    5: 'Automotive',
    6: 'Tools & Hardware',
    7: 'Baby Care',
    8: 'Mobiles & Accessories',
    9: 'Watches',
    10: 'Toys & School Supplies',
    11: 'Jewellery',
    12: 'Kitchen & Dining',
    13: 'Computers'
}

# gradio api
def predict_category(description):
    normalized_text = text_normalization(description)
    input_tfidf = TfidfVec.transform([normalized_text])
    prediction = tuned_SGD_model.predict(input_tfidf)
    return category_mapping[prediction[0]]

iface = gr.Interface(fn=predict_category, inputs="text", outputs="text", title="Ecommerce Product Categorization")
iface.launch()