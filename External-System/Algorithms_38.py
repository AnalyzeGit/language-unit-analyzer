#!/usr/bin/env python
# coding: utf-8

# In[60]:


# Option
import sys
import os
import json
import ijson
import csv
import tkinter as tk
import pandas as pd
import chardet
from openpyxl import Workbook
import time
from IPython.display import display
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np

# Natural Language processing 22 years old
from collections import Counter 
from konlpy.tag import Okt, Komoran, Hannanum, Kkma, Mecab

# Visualize 
from tkinter import filedialog, messagebox, ttk, StringVar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import font_manager, rc
from matplotlib.ticker import FuncFormatter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path


# In[61]:


# Action: 형태소 분석기 설정 

# 1. 단어 통계를 위한 Counter 객체 생성
word_counter = Counter()

# 2. 형태소 분석기(Mecab 로드)
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

# 3. 형태소 분석기 초기화
morpheme_analyzers = {
    "선택 없음": None,
    "Okt": Okt(),
    "Komoran": Komoran(),
    "Hannanum": Hannanum(),
    "Kkma": Kkma(),
    'Mecab': mecab }


# In[62]:


# Action: 폰트 설정

#if getattr(sys, 'frozen', False):  # 코드가 PyInstaller로 패키징된 경우
#    base_path = sys._MEIPASS
#else:
#    base_path = os.path.dirname(__file__)

base_path = os.getcwd()

font_path = os.path.join(base_path, "fonts", "malgun.ttf")
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


# In[63]:


# Action: ngrams 함수 생성

def generate_ngrams_prev_og(s, n):
    # Input: s = string, n = size of the ngram
    # Output: list of ngrams
    tokens = s.split()
    
    ngrams = zip(*[tokens[i:] for i in range(n)]) 
    
    n_grams_dataset = pd.DataFrame([" ".join(ngram) for ngram in ngrams],columns=['n_grams'])

    return n_grams_dataset


# In[64]:


def generate_ngrams_preva(s, n):
    tokens = s.split()
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i:i + n])
        ngrams.append(ngram)
    
    n_grams_dataset = pd.DataFrame(ngrams, columns=['n_grams'])
    return n_grams_dataset


# In[65]:


def apply_generate_ngrams(sentences,n):
    start_time = time.time()  # 함수 실행 전 시간 측정

    ngrams_comprehensions = [ngram for sentence in sentences for ngram in generate_ngrams(sentence, n)]

    #print(ngrams_comprehensions[:200])
    
    n_gram_dataset = pd.DataFrame(ngrams_comprehensions,columns=['n_grams']).reset_index().dropna()

    end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"엔 그램 함수_Execution time: {execution_time:.6f} seconds")

    return n_gram_dataset


# In[66]:


def apply_generate_ngrams_og(sentences,n):
    start_time = time.time()  # 함수 실행 전 시간 측정
    
    ngrams_list = []

    for sentence in sentences:
        #print(sentence)
        n_grams_sentence = generate_ngrams(sentence,n)
        
        ngrams_list.append(n_grams_sentence)

    n_gram_dataset = pd.concat(ngrams_list).reset_index().dropna()

    end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"엔 그램 함수_Execution time: {execution_time:.6f} seconds")

    return n_gram_dataset


# In[67]:


import pandas as pd

def generate_ngrams(s, n):
    tokens = s.split()
    
    n_grams_dataset = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
   
    return n_grams_dataset


# In[68]:


def apply_generate_ngrams_preva(sentences,n):
    ngrams_list = []

    # 병렬 처리
    ngram_result = Parallel(n_jobs=-1)(delayed(generate_ngrams)(sentence,n) for sentence in tqdm(sentences))
   
    n_gram_dataset = pd.concat(ngram_result).reset_index().dropna()
    
    return n_gram_dataset


# In[69]:


# Action: 콘코던스 단어 추적 함수 구현

def track_down_concordance_words():
    
    # concordance_entry에서 단어 목록을 가져옵니다.
    #concordance_words = concordance_entry.split('|')
    concordance_words_get = concordance_entry.get()
    concordance_words_get = replace_strip(concordance_words_get)
    
    if concordance_words_get != 'None':
        concordance_words = concordance_words_get.split('|')
        concordance_words = [word.strip() for word in concordance_words]

        return  concordance_words

    else:
        return 'None'


# In[70]:


# Action: 제외 단어 추적 함수 구현

def track_down_exclude_words():

    # 제외 단어 목록을 가져옵니다.
    exclude_words = exclude_words_entry.get().split('|')
    #exclude_words = exclude_words_entry.split('|')
    exclude_words = [word.strip() for word in exclude_words]

    return exclude_words


# In[71]:


def replace_strip(concordance_words_get):
    if concordance_words_get.strip() == '':
        return 'None'
    else:
        return concordance_words_get


# In[72]:


# Action: 콘코던스 필터링 함수 구현

def execute_concordence_sentence_only_fuction():
    start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 콘코던스 함수 실행
    concordance_words_get = track_down_concordance_words()
    # 문장 추출 함수 사용
    analyzed_folder,folder_path= extract_materials_be_analyzed() 
    # 폴더 경로 설정
    #folder_path = filedialog.askdirectory()

    concordence_dict = {}
    
    if (concordance_words_get) and ('None' not in concordance_words_get):
        #dict_with_concordance={'original': [], 'analyzed': []}
        list_with_concordence = [] 
        for key,value in analyzed_folder.items():
            all_sentences = value['Sentence']
            if any(any(con_word in sentence for con_word in concordance_words_get) for sentence in all_sentences):
                print("포함된 문장이 존재함")
                for con_word in concordance_words_get:
                    for sentence in all_sentences:
                        if con_word in sentence:
                            list_with_concordence.append(sentence)

                # 콘코던스 필러링 데이터 프레임화
                df_list_with_concordence = pd.DataFrame(list_with_concordence,columns=['Sentence'])
                # 콘코던스 딕셔너리 생성
                concordence_dict[key] = df_list_with_concordence 
            else:
                print("포함된 문장이 없습니다.")

        end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"콘코던스 함수_Execution time: {execution_time:.6f} seconds")

        return  concordence_dict,folder_path
    else:
        #print("콘코던스 키 입력하지 않음")
        #print(analyzed_folder)

        end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"콘코던스 함수_Execution time: {execution_time:.6f} seconds")
        
        return analyzed_folder,folder_path


# In[73]:


# Action: 문장 품사 태깅함수 구현

def tag_part_sentence():

    start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 분석 딕셔너리 
    sentence_dicts,folder_path = execute_concordence_sentence_only_fuction() 

    # 제외 단어 추척 함수 사용
    exclude_words_entry = track_down_exclude_words()
    
    # n_grams 생성 
    n_gram = ngram_cnt_entry.get()

    # 표 표출수 생성
    if extract_cnt_entry!=0:
        table_limit_count = int(extract_cnt_entry.get())

    # 형태소 분석기 생성
    morpheme_menu_get = morpheme_menu.get()

    # 아웃풋 딕셔너리 생성
    all_sentences = {'original': [], 'analyzed': []}
    
    for key,value in sentence_dicts.items():

        # 데이터 프레임 문장 추출
        sentences = value['Sentence']
        
        # 제외 단어 추척 함수 사용
        #exclude_words = track_down_exclude_words('먹었다')
    
        for sentence in sentences:
            # 선택한 형태소 분석기로 문장을 형태소 분석합니다.
            morphemes = morpheme_analyzers[morpheme_menu_get].pos(sentence)

            # 제외 단어 목록에 포함되지 않은 형태소만 추가합니다.
            filtered_morphemes = [f"{word}/{tag}" for word, tag in morphemes if word not in exclude_words_entry]

            # 문장을 형태소 분석된 형태로 변환합니다.
            analyzed_sentence = ' '.join(filtered_morphemes)

            # 기존 문장 저장
            all_sentences['original'].append(sentence)
    
            # 분석된 문장 저장
            all_sentences['analyzed'].append(analyzed_sentence)

        # n-gram을 생성합니다.
        #ngrams = apply_generate_ngrams(all_sentences['analyzed'], int(n_gram))

        # n_counts 생성
        #ngrams_count = count_n_grams(ngrams)

        # 분석 데이터 프레임 저장
        #pd.DataFrame(all_sentences).to_csv(f'{key}', index=False)
        #ngrams_count.to_csv(f'{key}_count.csv', index=False)

    # n-gram을 생성합니다.
    ngrams = apply_generate_ngrams(all_sentences['analyzed'], int(n_gram))

    # n_counts 생성
    ngrams_count = count_n_grams(ngrams)

    if table_limit_count!=0:
        ngrams_count = ngrams_count[:table_limit_count]

    else:
        pass

    pd.DataFrame(all_sentences).to_csv(f'{folder_path}\\analysis_table.csv' ,index=False)
    ngrams_count.to_csv(f'{folder_path}\\ngram_{n_gram}.csv', index=False)
    ###ngrams.to_csv(f'{folder_path}\\ngrams_dataset_{n_gram}.csv', index=False)

    # 네트워크 시각화 이미지 저장
    apply_network_chart(ngrams['n_grams'],all_sentences['analyzed'],folder_path)
    
    # 시각화 함수
    plot_data(ngrams_count)

    # 워드 클라우드 이미지 저장
    create_word_cloud(ngrams_count,folder_path)

    end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"태깅 함수_Execution time: {execution_time:.6f} seconds")

    return pd.DataFrame(all_sentences),folder_path


# In[74]:


# Action: 문장 품사 태깅함수 구현

# 형태소 분석 함수 정의
def analyze_sentence(sentence, exclude_words_entry):
    morpheme_analyzer = mecab  # 함수 내부에서 형태소 분석기 객체를 생성합니다.
    morphemes = morpheme_analyzer.pos(sentence)
    filtered_morphemes = [f"{word}/{tag}" for word, tag in morphemes if word not in exclude_words_entry]
    return sentence, ' '.join(filtered_morphemes)


def tag_part_sentence_preva():

    start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 분석 딕셔너리 
    sentence_dicts,folder_path = execute_concordence_sentence_only_fuction() 

    # 제외 단어 추척 함수 사용
    exclude_words_entry = track_down_exclude_words()
    
    # n_grams 생성 
    n_gram = ngram_cnt_entry.get()

    # 표 표출수 생성
    if extract_cnt_entry!=0:
        table_limit_count = int(extract_cnt_entry.get())

    # 형태소 분석기 생성
    morpheme_menu_get = morpheme_menu.get()

    # 아웃풋 딕셔너리 생성
    all_sentences = {'original': [], 'analyzed': []}
    
    # 데이터 프레임으로 변환
    dfs = []
    for file_path, df in sentence_dicts.items():
        #df['File'] = file_path
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    
    # 병렬 처리
    results = Parallel(n_jobs=-1)(delayed(analyze_sentence)(sentence, exclude_words_entry) for sentence in tqdm(data['Sentence']))

    #print(results)
    
    # 결과를 분할하여 저장
    for original, analyzed in results:
        all_sentences['original'].append(original)
        all_sentences['analyzed'].append(analyzed)

    # n-gram을 생성합니다.
    ngrams = apply_generate_ngrams(all_sentences['analyzed'], int(n_gram))

    # n_counts 생성
    ngrams_count = count_n_grams(ngrams)

    if table_limit_count!=0:
        ngrams_count = ngrams_count[:table_limit_count]

    else:
        pass

    pd.DataFrame(all_sentences).to_csv(f'{folder_path}\\analysis_table.csv' ,index=False)
    ngrams_count.to_csv(f'{folder_path}\\count.csv', index=False)
    
    # 시각화 함수
    plot_data(ngrams_count)

    # 워드 클라우드 이미지 저장
    create_word_cloud(ngrams_count,folder_path)

    end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"태깅 함수_Execution time: {execution_time:.6f} seconds")

    return pd.DataFrame(all_sentences),folder_path


# In[75]:


def analyze_sentences(sentence):
     # 제외 단어 추척 함수 사용
    exclude_words_entry = track_down_exclude_words()
    morpheme_analyzer = Okt()  # 함수 내부에서 형태소 분석기 객체를 생성합니다.
    morphemes = morpheme_analyzer.pos(sentence)
    filtered_morphemes = [f"{word}/{tag}" for word, tag in morphemes if word not in exclude_words_entry]
    return ' '.join(filtered_morphemes)
        
def tag_part_sentence_prevb():

    start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 분석 딕셔너리 
    sentence_dicts,folder_path = execute_concordence_sentence_only_fuction() 
    
    # 제외 단어 추척 함수 사용
    exclude_words_entry = track_down_exclude_words()
    
    # n_grams 생성 
    n_gram = ngram_cnt_entry.get()

    # 표 표출수 생성
    if extract_cnt_entry!=0:
        table_limit_count = int(extract_cnt_entry.get())

    # 데이터 프레임으로 변환
    dfs = []
    for file_path, df in sentence_dicts.items():
        dfs.append(df)

    all_sentences = pd.concat(dfs, ignore_index=True)
    
    # 데이터프레임의 apply 함수를 사용하여 벡터화 적용
    all_sentences['analyzed'] = all_sentences['Sentence'].apply(analyze_sentences)

    # n-gram을 생성합니다.
    ngrams = apply_generate_ngrams(all_sentences['analyzed'], int(n_gram))

    # n_counts 생성
    ngrams_count = count_n_grams(ngrams)

    if table_limit_count!=0:
        ngrams_count = ngrams_count[:table_limit_count]

    else:
        pass
    # 분석 데이터 프레임 저장
    #pd.DataFrame(all_sentences).to_csv(f'{key}', index=False)
    #ngrams_count.to_csv(f'{key}_count.csv', index=False)
        

    pd.DataFrame(all_sentences).to_csv(f'{folder_path}\\analysis_table.csv' ,index=False)
    ngrams_count.to_csv(f'{folder_path}\\count.csv', index=False)
    
    # 시각화 함수
    plot_data(ngrams_count)

    # 워드 클라우드 이미지 저장
    create_word_cloud(ngrams_count,folder_path)

    end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"태깅 함수_Execution time: {execution_time:.6f} seconds")

    return pd.DataFrame(all_sentences),folder_path


# In[76]:


def separate_sentence_to_phrase():

    start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 분석 딕셔너리 
    sentence_dicts,folder_path = execute_concordence_sentence_only_fuction() 
    
    # 제외 단어 추척 함수 사용
    exclude_words_entry = track_down_exclude_words()
    
    # n_grams 생성 
    n_gram = ngram_cnt_entry.get()

    # 표 표출수 생성
    if extract_cnt_entry!=0:
        table_limit_count = int(extract_cnt_entry.get())

    # 아웃풋 딕셔너리 생성
    all_sentences = {'original': [], 'analyzed': []}
    
    for key,value in sentence_dicts.items():
   
        # 데이터 프레임 문장 추출
        sentences = value['Sentence']
    
        for sentence in sentences:
            # 선택한 형태소 분석기로 문장을 형태소 분석합니다.
            morphemes = sentence.split(' ')
            # 제외 단어 목록에 포함되지 않은 형태소만 추가합니다.
            filtered_morphemes = [f"{word}" for word in morphemes if word not in exclude_words_entry]

            # 문장을 형태소 분석된 형태로 변환합니다.
            analyzed_sentence = ' '.join(filtered_morphemes)

            # 기존 문장 저장
            all_sentences['original'].append(sentence)
    
            # 분석된 문장 저장
            all_sentences['analyzed'].append(analyzed_sentence)

          
    # n-gram을 생성합니다.
    ngrams = apply_generate_ngrams(all_sentences['analyzed'], int(n_gram))

    # n_counts 생성
    ngrams_count = count_n_grams(ngrams)

    if table_limit_count!=0:
        ngrams_count = ngrams_count[:table_limit_count]

    else:
        pass
    # 분석 데이터 프레임 저장
    #pd.DataFrame(all_sentences).to_csv(f'{key}', index=False)
    #ngrams_count.to_csv(f'{key}_count.csv', index=False)
        

    pd.DataFrame(all_sentences).to_csv(f'{folder_path}\\analysis_table.csv' ,index=False)
    ngrams_count.to_csv(f'{folder_path}\\ngram_{n_gram}.csv', index=False)

    # 네트워크 시각화 이미지 저장
    apply_network_chart(ngrams['n_grams'],all_sentences['analyzed'],folder_path)

    # 시각화 함수
    plot_data(ngrams_count)

    # 워드 클라우드 이미지 저장
    create_word_cloud(ngrams_count,folder_path)

    end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"태깅 함수_Execution time: {execution_time:.6f} seconds")

    return pd.DataFrame(all_sentences),folder_path


# In[77]:


def split_sentence(sentence, exclude_words_entry):
    # sentence in sentences:
    # 선택한 형태소 분석기로 문장을 형태소 분석합니다.
    morphemes = sentence.split(' ')

    filtered_morphemes = [f"{word}" for word in morphemes if word not in exclude_words_entry]
    
    return sentence, ' '.join(filtered_morphemes)

def separate_sentence_to_phrase_preva():

    start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 분석 딕셔너리 
    sentence_dicts,folder_path = execute_concordence_sentence_only_fuction() 
    
    # 제외 단어 추척 함수 사용
    exclude_words_entry = track_down_exclude_words()
    
    # n_grams 생성 
    n_gram = ngram_cnt_entry.get()

    # 표 표출수 생성
    if extract_cnt_entry!=0:
        table_limit_count = int(extract_cnt_entry.get())

    # 아웃풋 딕셔너리 생성
    all_sentences = {'original': [], 'analyzed': []}
    
      
    # 데이터 프레임으로 변환
    dfs = []
    for file_path, df in sentence_dicts.items():
        df['File'] = file_path
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    
    # 데이터 프레임의 각 문장을 리스트로 변환
    data = []
    for sentences in df['Sentence']:
        for sentence in sentences:
            data.append((sentence, exclude_words_entry))

    # 병렬 처리
    results = Parallel(n_jobs=-1)(delayed(split_sentence)(sentence, exclude_words_entry) for sentence, exclude_words_entry in tqdm(data))

    # 결과를 분할하여 저장
    for original, analyzed in results:
        all_sentences['original'].append(original)
        all_sentences['analyzed'].append(analyzed)
          
    # n-gram을 생성합니다.
    ngrams = apply_generate_ngrams(all_sentences['analyzed'], int(n_gram))

    # n_counts 생성
    ngrams_count = count_n_grams(ngrams)

    if table_limit_count!=0:
        ngrams_count = ngrams_count[:table_limit_count]

    else:
        pass
    # 분석 데이터 프레임 저장
    #pd.DataFrame(all_sentences).to_csv(f'{key}', index=False)
    #ngrams_count.to_csv(f'{key}_count.csv', index=False)
        

    pd.DataFrame(all_sentences).to_csv(f'{folder_path}\\analysis_table.csv' ,index=False)
    ngrams_count.to_csv(f'{folder_path}\\count.csv', index=False)
    
    # 시각화 함수
    plot_data(ngrams_count)

    # 워드 클라우드 이미지 저장
    create_word_cloud(ngrams_count,folder_path)

    end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"태깅 함수_Execution time: {execution_time:.6f} seconds")

    return pd.DataFrame(all_sentences),folder_path


# In[78]:


def count_n_grams(words):
    # 단어 카운트
    word_count = Counter(words['n_grams'])
    
    word_count_data = pd.DataFrame(list(word_count.items()), columns=['Word','Frequency']).dropna()

    word_count_data = word_count_data.sort_values(by='Frequency', ascending=False)
    
    return word_count_data


# In[20]:


# Action: 제이슨 필터링 함수 구현

def filter_jason_folder(folder_path,filter_key,expected_value):
    """ 필터 값에 맞는 JASON 데이터 추출하기 

    파라미터: 폴더 경로, 필터 키, 필터 값

    반환 값: 필터 값에 맞는 JASON을 추가한 폴더 
    """

    start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 제이슨 폴더 생성
    jason_folder = []
    filter_jason_folder = []

    #print(f"현재 폴더:{folder_path}") 
    # 초기 값 설정
    cheked_file_path = None 
    #print("현재 폴더:",folder_path)
    for filename in os.listdir(folder_path):
        if (filename.endswith('.json')) | (filename.endswith('.JSON')):
            file_path = os.path.join(folder_path, filename)
            jason_folder.append(file_path)
            #print(f"필터링 폴더:{jason_folder}")
            #파일 인코딩 체크
            file_encoding = detect_encoding(file_path)
            with open(file_path, 'r', encoding=file_encoding) as file:
                data = json.load(file)
                # 필터 키 존재한다면
                if filter_key != None:
                    cheked_file_path = inspect_jason(data,filter_key,expected_value,file_path)
                # 만약 조건에 맞는 jason 파일이 있다면
                if cheked_file_path:
                    filter_jason_folder.append(cheked_file_path)
                    
    end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"폴더 필터링 함수_Execution time: {execution_time:.6f} seconds")

    # 필터링 제이슨 폴터가 존재한다면            
    if filter_jason_folder:
        #print("필터링 조건 통과")
        return filter_jason_folder
    else:
        #print("조건 통과하지 않음")
        return jason_folder


# In[41]:


# Action: filter key, values를 이용한 jason 파일 필터링 

def inspect_jason(jason_data,filter_key,expected_value,file_path):
    """
    필터 값과 일치하는 JASON 데이터 검사

    파라미터: 데이터, 필터 키, 필터 값, 데이터 경로

    반환 값: 필터 값에 맞는 JASON 데이터 경로
    """

    start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 필터 키 분리
    filter_key_list = filter_key.split('.')
    
    # 조건 실행
    try:
        for key in filter_key_list:
            if isinstance(jason_data,dict) and key in jason_data:
                jason_data = jason_data[key]
            elif isinstance(jason_data,list):
                 # 리스트의 경우, 리스트의 모든 요소를 포함하는 새 리스트를 생성
                jason_data = [subvalue[key] for subvalue in jason_data if key in subvalue]
            else:
                raise KeyError("Key not found in the JSON structure.")
        # 최종 값을 확인
        if isinstance(jason_data, list):
            matches = [val for val in jason_data if expected_value in val]
            if matches:
                print(f"Match found: {matches}")
                return file_path
            else:
                print("No match found.")
        else:
            if expected_value in jason_data:
                return file_path
                print(f"Match found: {jason_data}")
            else:
                print("No match found.")                

        end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"태그 필터링 함수_Execution time: {execution_time:.6f} seconds")

    except KeyError as e:
            print(f"Path not found in the JSON structure: {e}")

            end_time = time.time()  # 함수 실행 후 시간 측정
            execution_time = end_time - start_time  # 실행 시간 계산
            print(f"태그 필터링 함수_Execution time: {execution_time:.6f} seconds")

    except Exception as e:
            print(f"An error occurred: {e}") 


# In[42]:


# Action: 리스트 틀 정제 함수 

def flatten_list(nested_list):
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):  # 요소가 리스트인 경우, 재귀 호출
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)
    return flat_list


# In[43]:


# Action: Tag 내용 추출 함수 구현

def extract_tag(data,path_elements):
    """ JASON 데이터의 TAG 내용 추출
    
    파라미터: JASN 데이터, 테그 리스트

    반환 값: 테그 분석 내용    
    """
    # tag 원소를 담을 그릇
    tag_bowl = []
    last_element = path_elements[-1]
    
    try:
        # 첫 번째 경로 요소를 추출
        first_element = path_elements[0]
        # 현재 경로 요소가 리스트를 요구하는 경우
        if isinstance(data, list):
            if (first_element==last_element):
                for num in range(len(data)):
                    val = data[num][first_element]
                    tag_bowl.append(val)             
                return tag_bowl
            else:      
                # 리스트의 각 요소에 대해 재귀적으로 함수를 호출
                result = [extract_tag(item, path_elements) for item in data]
                # None 값을 제외한 결과만 필터링
                return [item for item in result if item is not None]       
        # 현재 데이터가 딕셔너리이고 경로 요소가 키로 존재하는 경우
        elif isinstance(data, dict) and first_element in data:      
            # 다음 경로 요소로 재귀적으로 함수를 호출    
            return extract_tag(data[first_element], path_elements[1:])      
        elif (first_element==last_element):
            return data[first_element]           
    except KeyError as e:
            print(f"Path not found in the JSON structure: {e}")
    except Exception as e:
            print(f"An error occurred: {e}") 


# In[44]:


# Action: 폴더 필터링, 분석 테그 내용 컴바인 함수 구현 

def extract_materials_be_analyzed():
    """ 폴더 필터링, 분석 태그 내용 컴바인 함수 구현

    반환 값: 테그 내용 리스트
    """
    # 태그 설정
    path_elements = user_input.get().split('.')
    
    if path_elements==[""]:
        messagebox.showinfo("Error","태그 값을 입력하세요")
        return   
    
    # 폴더 경로 설정
    folder_path = filedialog.askdirectory()
    #print("현자 경로:",folder_path)
    
    # 필터 키 설정
    filter_key = filter_key_entry.get()

    # 필터 값 설정
    tag_name = filter_value_entry.get()
    
    # 폴더에서 필터한 폴더를 반환
    fited_jason_forder = filter_jason_folder(folder_path,filter_key,tag_name)

    # 분석할 내용을 담을 딕셔너리 생성 
    analysis_bowl = {}

    # 폴더를 순회하면서 분석 내용 추출
    for jason_file in fited_jason_forder:

        # 파일 인코딩 체크
        file_encoding = detect_encoding(jason_file)
        
        with open(jason_file, 'r', encoding=file_encoding) as file:
            # 해당 데이터 
            data = json.load(file)
            # 파일 이름 생성 
            sentence_csv_path = create_file_name(folder_path,jason_file)
             # 내용 추출 
            rows_bowl = extract_tag(data,path_elements) 
            # 리스트 정제         
            clean_list = flatten_list(rows_bowl)
             # 데이터 프레임화
            clean_list = pd.DataFrame(clean_list,columns=['Sentence'])
            # 데이터 딕셔너리 추가
            analysis_bowl[sentence_csv_path] = clean_list

    return analysis_bowl,folder_path       


# In[45]:


# Action: 파일 이름 생성 함수

def create_file_name(folder_path,jason_file):
    
    # 파일 이름만 추출
    file_name_with_extension = os.path.basename(jason_file)

    # 확장자 제거
    file_name, _ = os.path.splitext(file_name_with_extension)
    
    sentence_csv_path = os.path.join(folder_path, f"{file_name}.csv")

    return sentence_csv_path


# In[46]:


# Action: Jason 인코딩 감지 함수 

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:  # 파일을 바이너리 모드로 열기
        raw_data = file.read(10000)  # 파일의 첫 부분을 읽어 인코딩 감지 (전체 파일을 읽어도 되지만 메모리를 많이 사용할 수 있음)
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    return encoding


# In[47]:


# Action: Window size 함수 

def cut_window_sizes(sentences,key):
    # window_size 설정
    window_size_get = int(window_cnt_entry.get())

    # 콘코던스 설정
    concordance_entry_get = track_down_concordance_words()
    
    if concordance_entry_get=='None':
        messagebox.showinfo("Information", "윈도우 사이즈를 사용하려면, 콘코던스 값을 입력해주세요")
        return 
        
    
    # 언어 단위 설정
    linguistic_unit_menu_get = linguistic_unit_menu.get()

    # 형태소 분석기 설정
    morpheme_menu_get = morpheme_menu.get()

    # 윈도우 사이즈 리스트 생성
    window_size_list = [] 
    
    # 문장 window_size 자르기
    for language_sentence in sentences:
        if linguistic_unit_menu_get == '어절':
            # 문장을 공백 기준으로 단어로 분리
            words = language_sentence.split()   
        else:
            words = morpheme_analyzers[morpheme_menu_get].morphs(language_sentence)
                                                                
        for concordance in concordance_entry_get:
            if concordance in language_sentence:
                for i, word in enumerate(words):
                    if concordance in word:
                        target_index = i
                        break
                    else:
                        pass  # 조건을 만족하지 않는 경우 아무 동작도 하지 않음

                # 널 값 제거 코드 추가
                #target_index = next((i for i, word in enumerate(words) if concordance in word),None)
                start_index = max(0, target_index - window_size_get)
                end_index = min(len(words), target_index + window_size_get + 1)
                window_sentence = ' '.join(words[start_index:end_index])
                window_size_list.append(window_sentence)
            else:
                pass              
                    
    window_size_dataset = pd.DataFrame(window_size_list,columns=['windowsize_sentence'])
    window_size_dataset = window_size_dataset.dropna()
    group_window_size_dataset = group_up(window_size_dataset)
    group_window_size_dataset.to_csv(f'{key}\\window_size.csv',index=False)


# In[48]:


def group_up(data):
    #group_windows = data.groupby('windowsize_sentence').count()
    group_windows = data['windowsize_sentence'].value_counts().reset_index()
    return group_windows


# In[49]:


def choose_language_unit():
    # 언어 단위 설정
    linguistic_unit_menu_get = linguistic_unit_menu.get()
    #linguistic_unit_menu_get = linguistic_unit_menu

    if linguistic_unit_menu_get =='어절':
        return separate_sentence_to_phrase()
    else:
        return tag_part_sentence()


# In[50]:


def select_final_language_analyzer():

    start_time = time.time()  # 함수 실행 전 시간 측정

    language_unit,fold_path = choose_language_unit()

    senetence_dataset = language_unit

    sentences = senetence_dataset['original']
    
    # 윈도우 사이즈 설정 
    window_size_get = window_cnt_entry.get()
    if window_size_get != '' :
        cut_window_sizes(sentences,fold_path)
        messagebox.showinfo("Information", "형태소 분석이 완료되었습니다! 결과가 저장되었습니다.")
        
        end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"Execution time: {execution_time:.6f} seconds")
        
        return language_unit
    else:
        messagebox.showinfo("Information", "형태소 분석이 완료되었습니다! 결과가 저장되었습니다.")
        
        end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"Execution time: {execution_time:.6f} seconds")
        
        return language_unit


# In[51]:


def only_numbers(char):
    return char.isdigit()

def update_user_input(*args):
    user_input.delete(0, 'end')
    user_input.insert(0, tag_options[tag_variable.get()])


# In[52]:


# Action: 스타일 설정

def configure_styles():
    style = ttk.Style()
    style.theme_use('clam')  # 클램 테마는 더 현대적인 느낌을 줍니다.
    style.configure('TLabel', font=('Arial', 10), background='white')
    style.configure('TEntry', font=('Arial', 10), padding=5)
    style.configure('TButton', font=('Arial', 10), padding=5)
    style.configure('TCombobox', font=('Arial', 10), padding=5)
    style.map('TCombobox', fieldbackground=[('readonly', 'white')],
              selectbackground=[('readonly', 'white')],
              selectforeground=[('readonly', 'black')])
    style.configure('TFrame', background='white')  # 프레임 배경색 설정
    style.configure('Horizontal.TProgressbar', background='#FA8072')


# In[53]:


def reset_table():
    global canvas, word_counter

    # Remove previous canvas if exists
    if canvas is not None:
        canvas.get_tk_widget().pack_forget()
        canvas = None

    # 표를 비웁니다.
    # result_tree.delete(*result_tree.get_children())

    # 단어 카운터를 초기화합니다.
    word_counter = Counter()

    # 프로그레스바를 0으로 초기화합니다.
    #progress_bar['value'] = 0


# In[54]:


def plot_data(ngrams_count):
    # 그래프를 그린 canvas 객체
    global canvas
    
    try:
        # Get the number of items to display from the entry widget
        number_of_items = int(graph_cnt_entry.get())
    except ValueError:  # In case of invalid input
        messagebox.showerror("Error", "그래프 표출수를 입력하세요")
        return

    # Create new figure
    fig = Figure(figsize=(8, 6), dpi=100)

    # Add a subplot to the new figure
    ax = fig.add_subplot(1, 1, 1)

    # Get the most common 'number_of_items' words
    common_words = ngrams_count[:number_of_items]

    # Separate the words and their counts
    words = list(common_words['Word'].values) 
    counts = list(common_words['Frequency'].values) 

    # Plot the data
    ax.bar(words, counts)

    # Adjust the x-axis labels
    ax.set_xticks(words)
    shortened_labels = [label if len(label) <= 10 else label[:10] + "..." for label in words]
    ax.set_xticklabels(shortened_labels, rotation=45, ha="right", fontsize=8)

    def hide_non_integers(x, pos):
        if x.is_integer():
            return "{:.0f}".format(x)
        return ""

    def integer_ticks(ax):
        # 현재 Y축의 눈금 위치를 가져옵니다.
        ticks = ax.get_yticks()

        # 소수점이 포함된 눈금 위치를 제거합니다.
        int_ticks = [tick for tick in ticks if tick.is_integer()]

        # Y축의 눈금 위치를 정수만 포함하도록 설정합니다.
        ax.set_yticks(int_ticks)

    # Set y-axis tick labels to show only integers
    ax.get_yaxis().set_major_formatter(FuncFormatter(hide_non_integers))
    integer_ticks(ax)

    fig.tight_layout()

    # Remove previous canvas if exists
    if canvas is not None:
        canvas.get_tk_widget().pack_forget()
        canvas = None

    # Create a new tkinter Canvas containing the figure
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()

    # Add the canvas to the Label widget
    canvas.get_tk_widget().pack() 


# In[55]:


#Action: 워크 들라우드 시각화

def create_word_cloud(data,file_path):

    base_path = os.getcwd()
    font_path = os.path.join(base_path, "fonts", "malgun.ttf")
    
    N = 20
    top_n_df = data.nlargest(N, 'Frequency')

    word_freq = dict(zip(top_n_df['Word'], top_n_df['Frequency']))
    
    # 워드 클라우드 생성
    wordcloud = WordCloud(width=800, height=400, background_color='white',font_path = font_path).generate_from_frequencies(word_freq)

    # 워드 클라우드 출력
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 축 제거
    plt.show()

    # 워드 클라우드 이미지 파일로 저장
    wordcloud.to_file(f'{file_path}\\wordcloud.png')


# In[56]:


def curved_edges(G, pos, ax, edge_colors, rad=0.2):
    for (u, v), color in zip(G.edges(), edge_colors):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        verts = [(x1, y1), ((x1 + x2) / 2, (y1 + y2) / 2 + rad), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor='none', edgecolor=color, lw=2)
        ax.add_patch(patch)


# In[57]:


def apply_network_chart(n_grams,sentences,file_path):
    
    num = 15

    nouns_counter = Counter(n_grams)

    n_gram_top_nouns = dict(nouns_counter.most_common(num))

    word2id = {w: i for i, w in enumerate(n_gram_top_nouns.keys())}

    id2word = {i: w for i, w in enumerate(n_gram_top_nouns.keys())}


    ngrams_adjMatrix = np.zeros((num,num), int)
    for sentence in sentences:
        for i, wi in id2word.items():
            if wi in sentence:
                for wj, j in word2id.items():
                    if i != j and wj in sentence:
                        ngrams_adjMatrix[i][j] += 1
    
    visualize_network_chart(ngrams_adjMatrix,id2word,file_path)


# In[58]:


def visualize_network_chart(ngrams_adjMatrix,labels,files_path):

    # numpy 배열을 그래프로 변환
    G = nx.from_numpy_array(ngrams_adjMatrix)

    # 노드 레이블 설정
    #labels = nx.get_node_attributes(G, 'label')

    # 엣지 가중치 설정
    #edge_labels = nx.get_edge_attributes(G, 'weight')

    # 엣지 색상과 투명도를 적절한 범위로 조정
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)   # 엣지 최대 웨이트 저장
    #norm = plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    edge_colors = [plt.cm.Reds (weight / max_weight) for weight in edge_weights]  # 색상 스케일 조정
    edge_alphas = [0.1 + (weight / max_weight) * 0.9 for weight in edge_weights]  # 투명도 스케일 조정
    
    #edge_colors = [plt.cm.plasma(norm(weight)) for weight in edge_weights]

    # 노드 색상 설정 (붉은 계열)
    # 노드 색상 설정 (낮은 값은 보라색, 높은 값은 빨간색)
    #norm = plt.Normalize(vmin=min(node_values), vmax=max(node_values))
    node_color = [plt.cm.Reds (np.random.rand()) for _ in range(len(G.nodes()))]
    #node_values = np.array([sum(edge_weights[v] for u, v in G.edges(node)) for node in G.nodes])
    #ode_color = [plt.cm.plasma(norm(value)) for value in node_values]
    
    # 그래프 시각화
    pos = nx.spring_layout(G)
    #plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(12, 8))

    # 노드 그리기
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_color,
        node_size=200,
        ax=ax)
    
    # 노드 라벨 그리기
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=12,
        font_color='black',
        font_family=font_name,
        ax=ax)

    # 엣지 그리기
    for (u, v), color, alpha in zip(G.edges(), edge_colors,edge_alphas):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3, edge_color=[color], alpha=alpha)

    #plt.title('Network Graph with Adjusted Edge Colors and Node Labels')
    #plt.show()
    
    # 엣지 그리기 (곡선으로)
    curved_edges(G, pos, ax, edge_colors)
    
    # 네크워크 시각화 이미지 파일로 저장+
    plt.savefig(f'{files_path}\\network.png')


# In[59]:


# Action: GUI 생성

root = tk.Tk()
root.title("n-gram 및 형태소 분석기 v1.1")
# 프로그램의 고정된 크기
program_width = 670
program_height = 850

# 화면의 중앙에 프로그램이 위치하도록 좌표를 계산합니다.
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_x = int((screen_width - program_width) / 2)
center_y = int((screen_height - program_height) / 2)

# 프로그램의 위치와 크기를 설정합니다.
root.geometry(f'{program_width}x{program_height}+{center_x}+{center_y}')

configure_styles()

# 숫자 입력 확인을 위한 유효성 검사 커맨드 생성
vcmd = root.register(only_numbers)

# (1,1)
tag_label = tk.Label(root, text="* 태그", anchor='w')
tag_label.grid(row=0, column=0, sticky='we', padx=10, pady=5)

# (1,2)
user_input = ttk.Entry(root)
user_input.grid(row=0, column=1, sticky='ew', padx=10, pady=5)

# (1,3)
tag_menu_label = tk.Label(root, text="* 태그 선택", anchor='w')
tag_menu_label.grid(row=0, column=2, sticky='we', padx=10, pady=5)

# (1,4)
tag_variable = StringVar(root)
tag_variable.trace("w", update_user_input)
tag_options = {
    "신문 말뭉치": "document.paragraph.form",
    "일상 대화 말뭉치": "document.utterance.form",
    "직접 입력": "",
}
tag_menu = ttk.Combobox(root, textvariable=tag_variable, values=list(tag_options.keys()), state='readonly')
tag_menu.grid(row=0, column=3, sticky='ew', padx=10, pady=5)
tag_menu.set("직접 입력")

# (2,1)
ngram_cnt_label = tk.Label(root, text="* n-gram 사이즈", anchor='w')
ngram_cnt_label.grid(row=1, column=0, sticky='we', padx=10, pady=5)

# (2,2)
ngram_cnt_entry = ttk.Entry(root, validate="key", validatecommand=(vcmd, '%S'))
ngram_cnt_entry.grid(row=1, column=1, sticky='ew', padx=10, pady=5)
ngram_cnt_entry.insert(0, 2)

# (2,3)
tag_menu_label = tk.Label(root, text="* 형태소 분석기", anchor='w')
tag_menu_label.grid(row=1, column=2, sticky='we', padx=10, pady=5)

# (2,4)
morpheme_analyzer = StringVar(root)
morpheme_menu = ttk.Combobox(root, textvariable=morpheme_analyzer, values=list(morpheme_analyzers.keys()), state='readonly')
morpheme_menu.grid(row=1, column=3, sticky='ew', padx=10, pady=5)
morpheme_menu.set("Mecab")

# (3,1)
filter_key_label = tk.Label(root, text="필터키", anchor='w')
filter_key_label.grid(row=2, column=0, sticky='we', padx=10, pady=5)

# (3,2)
filter_key_entry = ttk.Entry(root)
filter_key_entry.grid(row=2, column=1, sticky='ew', padx=10, pady=5)

# (3,3)
tag_menu_label = tk.Label(root, text="필터값", anchor='w')
tag_menu_label.grid(row=2, column=2, sticky='we', padx=10, pady=5)

# (3,4)
filter_value_entry = ttk.Entry(root)
filter_value_entry.grid(row=2, column=3, sticky='ew', padx=10, pady=5)

# (4,1)
concordance_label = tk.Label(root, text="콘코던스 단어(|로 구분)", anchor='w')
concordance_label.grid(row=3, column=0, sticky='we', padx=10, pady=5)

# (4,2)
concordance_entry = ttk.Entry(root)
concordance_entry.grid(row=3, column=1, sticky='ew', padx=10, pady=5)

# (4,3)
exclude_words_label = tk.Label(root, text="제외 단어(|로 구분)", anchor='w')
exclude_words_label.grid(row=3, column=2, sticky='we', padx=10, pady=5)

# (4,4)
exclude_words_entry = ttk.Entry(root)
exclude_words_entry.grid(row=3, column=3, sticky='ew', padx=10, pady=5)

# (5,1)
window_cnt_label = tk.Label(root, text="window 사이즈", anchor='w')
window_cnt_label.grid(row=4, column=0, sticky='we', padx=10, pady=5)

# (5,2)
window_cnt_entry = ttk.Entry(root, validate="key", validatecommand=(vcmd, '%S'))
window_cnt_entry.grid(row=4, column=1, sticky='ew', padx=10, pady=5)
window_cnt_entry.insert(0, "")

# (5,3)
linguistic_unit_label = tk.Label(root, text="* 언어 단위 선택", anchor='w')
linguistic_unit_label.grid(row=4, column=2, sticky='we', padx=10, pady=5)

# (5,4)
linguistic_unit_variable = StringVar(root)
linguistic_unit_variable.trace("w", update_user_input)
linguistic_unit_options = {
    "형태소": "",
    "어절": ""}
linguistic_unit_menu = ttk.Combobox(root, textvariable=linguistic_unit_variable, values=list(linguistic_unit_options.keys()), state='readonly')
linguistic_unit_menu.grid(row=4, column=3, sticky='ew', padx=10, pady=5)
linguistic_unit_menu.set("형태소")

# (6,1)
graph_cnt_label = tk.Label(root, text="* 그래프 표출수(최대20)", anchor='w')
graph_cnt_label.grid(row=5, column=0, sticky='we', padx=10, pady=5)

#(6,2)
graph_cnt_entry = ttk.Entry(root, validate="key", validatecommand=(vcmd, '%S'))
graph_cnt_entry.grid(row=5, column=1, sticky='ew', padx=10, pady=5)
graph_cnt_entry.insert(0, "10")

# (6,3)
extract_cnt_label = tk.Label(root, text="표 표출수", anchor='w')
extract_cnt_label.grid(row=5, column=2, sticky='we', padx=10, pady=5)

# (6,4)
extract_cnt_entry = ttk.Entry(root, validate="key", validatecommand=(vcmd, '%S'))
extract_cnt_entry.grid(row=5, column=3, sticky='ew', padx=10, pady=5)
extract_cnt_entry.insert(0,0)

# (7,1)
folder_button = ttk.Button(root, text="폴더 선택 및 분석", command=select_final_language_analyzer)
folder_button.grid(row=6, column=0, sticky='ew', padx=10, pady=5, columnspan=2)

# (7,2)
reset_button = ttk.Button(root, text="결과창 리셋", command=reset_table)
reset_button.grid(row=6, column=2, sticky='ew', padx=10, pady=5, columnspan=2)

# (8,1)
graph_frame = ttk.Frame(root, height=450)  # height를 설정해 줍니다.
graph_frame.grid(row=7, column=0, columnspan=4, sticky='ew')  # sticky를 'ew'로 변경합니다.

root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(6, weight=1)
root.grid_rowconfigure(7, weight=1)

# 초기 캔버스 설정을 None으로 합니다. 'embed_figure' 함수 정의가 필요합니다.
canvas = None

root.mainloop()


# In[40]:


#Action: 워크 들라우드 시각화

def create_word_cloud(data,file_path):

    base_path = os.getcwd()
    font_path = os.path.join(base_path, "fonts", "malgun.ttf")
    
    N = 20
    top_n_df = data.nlargest(N, 'Frequency')

    word_freq = dict(zip(top_n_df['Word'], top_n_df['Frequency']))
    
    # 워드 클라우드 생성
    wordcloud = WordCloud(width=800, height=400, background_color='white',font_path = font_path).generate_from_frequencies(word_freq)

    # 워드 클라우드 출력
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 축 제거
    plt.show()

    # 워드 클라우드 이미지 파일로 저장+
    wordcloud.to_file(f'{file_path}\\wordcloud.png')

