from tqdm import tqdm
import csv
import pandas as pd

import re
import math
import string

def get_words_from_text(text): 
    translation_table = str.maketrans(string.punctuation+string.ascii_uppercase,
                                    " "*len(string.punctuation)+string.ascii_lowercase)
    text = re.sub(r'\d\.(\s)+', '', text)
    text = text.translate(translation_table)
    word_list = text.split()
        
    return word_list
    
    
    # counts frequency of each word
    # returns a dictionary which maps
    # the words to  their frequency.
def count_frequency(word_list, stopwords): 
        
    D = {}
        
    for new_word in word_list:
        if stopwords and new_word in stopwords:
            continue
        if new_word in D:
            D[new_word] = D[new_word] + 1
                
        else:
            D[new_word] = 1
                
    return D
    
    # returns dictionary of (word, frequency)
    # pairs from the previous dictionary.
def word_frequencies(text, stopwords): 
        
    word_list = get_words_from_text(text)
    freq_mapping = count_frequency(word_list, stopwords)
    
    return freq_mapping
    
    
    # returns the dot product of two documents
def dotProduct(D1, D2): 
    Sum = 0.0
        
    for key in D1:
            
        if key in D2:
            Sum += (D1[key] * D2[key])
                
    return Sum
    
    # returns the angle in radians 
    # between document vectors
def vector_angle(D1, D2): 
    numerator = dotProduct(D1, D2)
    denominator = math.sqrt(dotProduct(D1, D1)*dotProduct(D2, D2))
        
    return math.acos(numerator / denominator)
    
    
def documentSimilarity(text_1, text_2, stopwords= None):
        
    # filename_1 = sys.argv[1]
    # filename_2 = sys.argv[2]
    sorted_word_list_1 = word_frequencies(text_1, stopwords)
    sorted_word_list_2 = word_frequencies(text_2, stopwords)
    distance = vector_angle(sorted_word_list_1, sorted_word_list_2)
        
    return distance

def create_mapping(data, desc):
    scores = []
    src = []
    dest = []
    for i in tqdm(range(len(data)), position= 0, desc= desc):
        if len(data['silver_rationales'][i]) == 0:
            continue
        doc_1 = [data['facts'][i][idx] for idx in data['silver_rationales'][i]]
        for j in tqdm(range(i + 1, len(data)), position= 1, leave= False):
            if len(data['silver_rationales'][j]) == 0:
                continue
            doc_2 = [data['facts'][j][idx] for idx in data['silver_rationales'][j]]
            total_score = []
            for doc1 in doc_1:
                score = []
                for doc2 in doc_2:
                    s = documentSimilarity(doc1, doc2)
                    score.append(s)
                total_score.append(sum(score) / len(doc_2))
            scores.append(sum(total_score) / len(doc_1))
            src.append(data['case_id'][i])
            dest.append(data['case_id'][j])

    pd.DataFrame({
        'src': src,
        'dest': dest,
        'score': scores
    }).to_csv(f'./data/{desc}_mapping.csv', index= False)

if __name__ == '__main__':
    train_data = pd.read_json('./data/train.jsonl', lines= True)
    val_data = pd.read_json('./data/dev.jsonl', lines= True)
    test_data = pd.read_json('./data/test.jsonl', lines= True)

    create_mapping(train_data, 'training_data')
    create_mapping(val_data, 'validation_data')
    create_mapping(test_data, 'test_data')
