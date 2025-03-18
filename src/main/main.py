import re
import elasticsearch
import fasttext
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
import pandas as pd

# load the saved classifier and vectorizer
with open('classifier/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('classifier/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# question file path
question_file = "test_sel_blacked.txt"
file1 = open(question_file, 'r', encoding='utf-8')
Lines = file1.readlines()

# elasticsearch
es = elasticsearch.Elasticsearch("http://localhost:9200")
index_name = "wikibase_index"

# load FastText model
ft = fasttext.load_model('cc.en.300.bin')

# load spacy model (python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")


# lowercase, remove punctuation and stop words
def normalize_question(sentence):
    doc = nlp(sentence)
    normalized_tokens = [token.text.lower() for token in doc if not (token.is_punct or token.is_stop)]
    normalized_sentence = ' '.join(normalized_tokens)
    return normalized_sentence


# elasticsearch query; returns 3 documents
def search_answers(user_query):
    query = {
        "query": {
            "multi_match": {
                "query": normalize_question(user_query),
                "fields": ["text^3.0", "title"],
                "type": "cross_fields",
                "tie_breaker": 0.3,
                "operator": "or"
            }
        },
        "size": 3
    }
    result = es.search(index=index_name, body=query)
    return result["hits"]["hits"]


# fasttext
def get_sentence_vectors_answer(text, model):
    clean_text = text.replace('\n', '')
    vector = model.get_sentence_vector(clean_text)
    return vector


# fasttext
def get_sentence_vectors_question(text, model):
    cleaned_text = text.replace('\n', '')
    cleaned_text = re.sub(r'[:.?!\'"]', '', cleaned_text)
    vector = model.get_sentence_vector(cleaned_text)
    return vector


def predict_question_type(question):
    x_test_vectors = vectorizer.transform([question])
    y_pred = classifier.predict(x_test_vectors)
    return y_pred


def extract_answer_word(sentence, answer_type, question):
    doc_question = nlp(question)
    words_in_question = [token.text for token in doc_question if token.ent_type_ != 0]

    doc_sentence = nlp(sentence)

    # empty list to store answer words
    extracted_words = []

    # depending on the answer type, extract answer word with spacy NER
    answer_type_str = answer_type[0] if isinstance(answer_type, list) else answer_type
    if "HUM:ind" in answer_type_str:
        persons = [ent.text for ent in doc_sentence.ents if ent.label_ == "PERSON"]
        extracted_words.extend(persons)
    elif "HUM:gr" in answer_type_str:
        group = [ent.text for ent in doc_sentence.ents if ent.label_ in ["NORP", "ORG"]]
        extracted_words.extend(group)
    elif "LOC" in answer_type_str:
        locations = [ent.text for ent in doc_sentence.ents if ent.label_ in ["FAC", "GPE", "LOC"]]
        extracted_words.extend(locations)
    elif "NUM:date" in answer_type_str:
        date = [ent.text for ent in doc_sentence.ents if ent.label_ == "DATE"]
        extracted_words.extend(date)
    elif "NUM:count" in answer_type_str:
        date_entities = [ent.text for ent in doc_sentence.ents if ent.label_ == "DATE"]
        if date_entities and ' m.' in sentence:
            # use regex to find the number before ' m.' (m. = meter)
            match = re.search(r'\b(\d+)\s*m\.', sentence)
            if match:
                number_before_m = match.group(1)
                extracted_words.append(number_before_m + ' m.')
        else:
            numbers = [ent.text for ent in doc_sentence.ents if ent.label_ in ["CARDINAL", "PERCENT", "QUANTITY", "MONEY"]]
            extracted_words.extend(numbers)
    elif "ENTY:cremat" in answer_type_str:
        creative = [ent.text for ent in doc_sentence.ents if ent.label_ in ["WORK_OF_ART", "LAW"]]
        extracted_words.extend(creative)
    elif "ENTY:animal" or "ENTY:food" in answer_type_str:
        # idea here is to extract recognized entites and remove them from sentence
        # then take remaining noun as answer word
        all_ent = [ent.text for ent in doc_sentence.ents]
        words = sentence.split()
        filtered_words = [word for word in words if word not in all_ent]
        new_sentence = ' '.join(filtered_words)
        docu = nlp(new_sentence)
        nouns = [token.text for token in docu if token.pos_ == "NOUN"]
        extracted_words.extend(nouns)
    elif "ENTY" in answer_type_str:
        entities = [ent.text for ent in doc_sentence.ents]
        extracted_words.extend(entities)

    # subtract words in the question from the extracted words
    extracted_words = [word for word in extracted_words if word not in words_in_question]

    # return the first non-question word as the answer word
    return extracted_words[0] if extracted_words else None


def get_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


# question loop
for i, question in enumerate(Lines):
    # print(f"Question: {question}")
    results = search_answers(question)
    answer_type = predict_question_type(question)
    answer_list = []
    # print(answer_type)

    # answer loop
    for j, hit in enumerate(results):
        doc = hit["_source"]
        question_vectors = get_sentence_vectors_question(question, ft)
        question_vectors_normalized = normalize([question_vectors])

        sentences = get_sentences(doc["text"])

        # taking from doc1-> 5, doc2-> 3, doc3-> 2 sentences
        num_sentences_to_select = 5 if j == 0 else (3 if j == 1 else 2)

        max_similarity_scores = []
        best_sentences = []

        for k, sentence in enumerate(sentences):
            if k >= num_sentences_to_select:
                break

            sentence_vectors = get_sentence_vectors_answer(sentence, ft)
            sentence_vectors_normalized = normalize([sentence_vectors])
            cosine_similarity_score = cosine_similarity(question_vectors_normalized, sentence_vectors_normalized)

            max_similarity_scores.append(np.max(cosine_similarity_score))
            best_sentences.append(sentence.strip())

        # sort sentences based on similarity scores
        sorted_indices = np.argsort(max_similarity_scores)[::-1]

        # retrieving word from sentence
        for idx in sorted_indices:
            # print(f"Sentence:{best_sentences[idx]}")
            answer_word = extract_answer_word(best_sentences[idx], answer_type[0], question)
            if answer_word is None:
                answer_list.append('None')
            else:
                answer_list.append(answer_word)
            # print(f"{answer_word}")
    df = pd.DataFrame([answer_list])
    with open('test_sample_answers.csv', mode='a', newline='', encoding='utf-8') as file:
        df.to_csv(file, index=False, header=False, sep=';')
