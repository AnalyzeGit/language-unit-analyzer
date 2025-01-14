# Handling
import sys
import os
import json
import tkinter as tk
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import chardet

# Natural Language Processing
from collections import Counter 
from konlpy.tag import Okt, Komoran, Hannanum, Kkma, Mecab

# Visualize and System
from tkinter import filedialog, messagebox, ttk, StringVar
from matplotlib.path import Path

# Module
from loadDatabase import *


# In[2]:


# Action: 형태소 분석기 설정 

# 1. 단어 통계를 위한 Counter 객체 생성
word_counter = Counter()

# 2. 형태소 분석기(Mecab 로드)
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

# 3. 형태소 분석기 초기화
morpheme_analyzers = {
    #"Okt": Okt(),
    #"Komoran": Komoran(),
    #"Hannanum": Hannanum(),
    #"Kkma": Kkma(),
    'Mecab': mecab }


# In[3]:


def generate_ngrams(s, n):
    tokens = s.split()
    
    n_grams_dataset = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
   
    return n_grams_dataset


# In[4]:


def apply_generate_ngrams(sentences,n):
    ngram_start_time = time.time()  # 함수 실행 전 시간 측정

    ngrams_comprehensions = [ngram for sentence in sentences['analyzed'] for ngram in generate_ngrams(sentence, n)]
    
    n_gram_dataset = pd.DataFrame(ngrams_comprehensions,columns=['n_grams']).reset_index()
    n_gram_dataset['date'] = sentences['date']
    n_gram_dataset['topic'] = sentences['topic']

    n_gram_dataset = n_gram_dataset.dropna()

    ngram_end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = ngram_end_time - ngram_start_time  # 실행 시간 계산
    print(f"엔 그램 함수_Execution time: {execution_time:.6f} seconds")

    return n_gram_dataset


# In[5]:


# Action: 문장 품사 태깅함수 구현

def tag_part_sentence_pre():

    tag_start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 분석 딕셔너리 
    sentence_datas,folder_path = extract_materials_be_analyzed() 
    
    # 어절 n_grams 생성
    separate_sentence_to_phrase(sentence_datas,folder_path)

    # n_grams 생성 
    n_grams = [1,2,3]

    # 아웃풋 딕셔너리 생성
    all_sentences = {'original': [], 'analyzed': []}

    for morpheme_name in morpheme_analyzers.keys():

        morphem_repetition_start_time = time.time() 
        
        sentences = sentence_datas['Sentence']
        
        for sentence in tqdm(sentences):
          
            # 선택한 형태소 분석기로 문장을 형태소 분석합니다.
            morphemes = morpheme_analyzers[morpheme_name].pos(sentence)

            # 제외 단어 목록에 포함되지 않은 형태소만 추가합니다.
            filtered_morphemes = [f"{word}/{tag}" for word, tag in morphemes]

            # 문장을 형태소 분석된 형태로 변환합니다.
            analyzed_sentence = ' '.join(filtered_morphemes)

            # 기존 문장 저장
            all_sentences['original'].append(sentence)
    
             # 분석된 문장 저장
            all_sentences['analyzed'].append(analyzed_sentence)
        
        # n-gram을 생성합니다.
        for n_gram in n_grams:
            
            ngrams = apply_generate_ngrams(all_sentences['analyzed'], int(n_gram))

            # n_counts 생성
            ngrams_count = count_n_grams(ngrams)

            ngrams_count.to_csv(f'{folder_path}\\morpheme_{morpheme_name}_ngram_{n_gram}.csv', index=False)

        morphem_repetition_end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = morphem_repetition_end_time - morphem_repetition_start_time  # 실행 시간 계산
        print(f"{morpheme_name}_반복문_Execution time: {execution_time:.6f} seconds")
    
    tag_end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = tag_end_time - tag_start_time  # 실행 시간 계산
    print(f"형태소 + 어절 함수_Execution time: {execution_time:.6f} seconds")


# In[6]:


# Action: 문장 품사 태깅함수 구현

# 형태소 분석 및 필터링 함수 정의
def analyze_sentence(sentence, morpheme_analyzer):
    morphemes = morpheme_analyzer.pos(sentence)
    filtered_morphemes = [f"{word}/{tag}" for word, tag in morphemes]
    return ' '.join(filtered_morphemes)


def tag_part_sentence():

    morphem_dataset = []

    tag_start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 분석 딕셔너리 
    sentence_datas,folder_path,middle_path = extract_materials_be_analyzed() 
    
    # 어절 n_grams 생성
    phrase_dataset = separate_sentence_to_phrase(sentence_datas,folder_path,middle_path)

    # n_grams 생성 
    n_grams = [1,2,3]

    for morpheme_name in morpheme_analyzers.keys():
        # 아웃풋 딕셔너리 생성
        all_sentences = {'date':[],'topic':[],'analyzed':[]}
        
        # 형태소 분석기 인스턴스 가져오기
        morpheme_analyzer = morpheme_analyzers[morpheme_name]

        morphem_repetition_start_time = time.time() 
        
        for index,row in tqdm(sentence_datas.iterrows()):
            
            date = row['date']
            topic = row['topic']
            sentences = row['sentences']

            for sentence in sentences:

                analyzed_sentence = analyze_sentence(sentence, morpheme_analyzer)

                # 기존 문장 저장
                all_sentences['analyzed'].append(analyzed_sentence)
                # date, topic 설정
                all_sentences['date'].append(date)
                all_sentences['topic'].append(topic)

        all_sentences = pd.DataFrame(all_sentences)   

        # n_counts 생성
        output_dataset = group_up(all_sentences,folder_path,morpheme_name,middle_path)

        output_dataset_concat = pd.concat(output_dataset)

        morphem_dataset.append(output_dataset_concat)

        morphem_repetition_end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = morphem_repetition_end_time - morphem_repetition_start_time  # 실행 시간 계산
        print(f"{morpheme_name}_반복문_Execution time: {execution_time:.6f} seconds")


    morphem_dataset_concat = pd.concat(morphem_dataset)

    final_concat = pd.concat([phrase_dataset,morphem_dataset_concat],ignore_index=True)
        
    final_concat = final_concat.reset_index()

    final_concat.columns=['STAT_NGRAM_NO','CORPUS_CL','ANALS_YEAR','ANALS_TOPIC','MORPHEME','NGRAM_TY','NGRAM_ORDR','NGRAM_FREQUENCY','NGRAM_VAL']

    #morphem_dataset_concat.to_csv(f"{folder_path}\\pharse_num.csv",index=False)
    #phrase_dataset.to_csv(f"{folder_path}\\morphems_num.csv",index=False)
    #final_concat.to_csv(f"{folder_path}\\final_concat.csv",index=False)
    load_database(final_concat,'corpus_stat_ngram_pra')

    
    tag_end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = tag_end_time - tag_start_time  # 실행 시간 계산
    print(f"형태소 + 어절 함수_Execution time: {execution_time:.6f} seconds")


# In[7]:


def separate_sentence_to_phrase(sentence_datas,folder_path,middle_path):

    phrase_start_time = time.time()  # 함수 실행 전 시간 측정

    # 아웃풋 딕셔너리 생성
    all_sentences = {'date':[],'topic':[],'analyzed':[]}
    
    for index,row in tqdm(sentence_datas.iterrows()):
        date = row['date']
        topic = row['topic']
        sentences = row['sentences']

        for sentence in sentences:
                
            # 선택한 형태소 분석기로 문장을 형태소 분석합니다.
            morphemes = sentence.split(' ')
            # 제외 단어 목록에 포함되지 않은 형태소만 추가합니다.
            filtered_morphemes = [f"{word}" for word in morphemes]

            # 문장을 형태소 분석된 형태로 변환합니다.
            analyzed_sentence = ' '.join(filtered_morphemes)
            
            all_sentences['analyzed'].append(analyzed_sentence)
            all_sentences['date'].append(date)
            all_sentences['topic'].append(topic)

    all_sentences = pd.DataFrame(all_sentences)

    # n_counts 생성
    phrase_data = group_up(all_sentences,folder_path,None,middle_path)

    phrase = pd.concat(phrase_data)
    
    phrase_end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = phrase_end_time - phrase_start_time  # 실행 시간 계산
    print(f"어절 함수_Execution time: {execution_time:.6f} seconds")

    return phrase


# In[8]:


def group_up(data,folder_path,morpheme_name,middle_path):

    phrase_data = []

    dates = data['date'].unique()
    topics = data['topic'].unique()

    # n_grams 생성 
    n_grams = [1,2,3]

    for n_gram in n_grams:
        for date in dates:
            for topic in topics:
                
                selected_sentences = data[(data['date']==date) & (data['topic']==topic)]

                # n-gram을 생성합니다.
                ngrams = apply_generate_ngrams(selected_sentences, int(n_gram))
            
                ngrams_count = count_n_grams(ngrams)

                ouput_dataset = clean_up_dataframes(ngrams_count,date,topic,morpheme_name,n_gram,middle_path)

                ouput_dataset = ouput_dataset[['CORPUS_CL','ANALS_YEAR','ANALS_TOPIC','MORPHEME','NGRAM_TY','NGRAM_ORDR','NGRAM_FREQUENCY','NGRAM_VAL']]

                phrase_data.append(ouput_dataset)
                
            #ngrams_count.to_csv(f"{folder_path}\\test_num_{num}.csv",index=False)

    return phrase_data              


# In[9]:


def count_n_grams(words):
    # 단어 카운트
    word_count = Counter(words['n_grams'])

    # 데이터 프레임 
    word_count_data = pd.DataFrame(list(word_count.items()), columns=['NGRAM_VAL','NGRAM_FREQUENCY']).dropna()

    # 순서정렬
    word_count_data = word_count_data.sort_values(by='NGRAM_FREQUENCY', ascending=False)[:200]

    # 인덱스 제거
    word_count_data = word_count_data.reset_index(drop=True)

    # 순서 인덱스 생성
    word_count_data = word_count_data.reset_index()

    word_count_data['index'] = word_count_data['index'] + 1

    # 컬럼 이름 변경
    word_count_data.columns = ['NGRAM_ORDR','NGRAM_VAL','NGRAM_FREQUENCY']
    
    return word_count_data


# In[10]:


def clean_up_dataframes(ngrams_count,date,topic,morpheme_name,n_gram,middle_path):

    # 변수 생성
    ngrams_count['ANALS_YEAR'] = date
    ngrams_count['ANALS_TOPIC'] = topic
    ngrams_count['NGRAM_TY'] = n_gram


    # 말뭉치 종류 생성
    if middle_path!='utterance':
        ngrams_count['CORPUS_CL'] = 'newspaper'
    else:
        ngrams_count['CORPUS_CL'] = 'dialogue'
    
    # 형태소 변수 생성
    if morpheme_name:
        ngrams_count['MORPHEME'] = morpheme_name
    else:
        ngrams_count['MORPHEME'] = 'pharse'

    return ngrams_count


# In[11]:


# Action: 제이슨 필터링 함수 구현

def filter_jason_folder(folder_path):
    """ 필터 값에 맞는 JASON 데이터 추출하기 

    파라미터: 폴더 경로, 필터 키, 필터 값

    반환 값: 필터 값에 맞는 JASON을 추가한 폴더 
    """

    start_time = time.time()  # 함수 실행 전 시간 측정
    
    # 제이슨 폴더 생성
    jason_folder = []

    # 초기 값 설정
    cheked_file_path = None 
    for filename in os.listdir(folder_path):
        if (filename.endswith('.json')) | (filename.endswith('.JSON')):
            file_path = os.path.join(folder_path, filename)
            jason_folder.append(file_path)

    end_time = time.time()  # 함수 실행 후 시간 측정
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"폴더 필터링 함수_Execution time: {execution_time:.6f} seconds")

    return jason_folder


# In[12]:


# Action: 리스트 틀 정제 함수 

def flatten_list(nested_list):
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):  # 요소가 리스트인 경우, 재귀 호출
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)
    return flat_list


# In[13]:


# Action: Tag 내용 추출 함수 구현

def extract_tag_pre(data,path_elements):
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


# In[14]:


# Action: Tag 내용 추출 함수 구현
# development_progress
def extract_tag(data,path_elements):
    """ JASON 데이터의 TAG 내용 추출
    
    파라미터: JASN 데이터, 테그 리스트

    반환 값: 테그 분석 내용    
    """
    # tag 원소를 담을 그릇
    tag_bowl = []
    start_path_element = path_elements[0]
    middle_path_element = path_elements[1]
    end_path_element = path_elements[2]
    
    for doc in data[start_path_element]:

        doct_dict = {'date':'','topic':'','sentences':[]}

        date = doc['metadata']['date']
        topic = doc['metadata']['topic']

        if middle_path_element !='utterance':
            paragraphs =  doc['paragraph']
        else:
            paragraphs = doc['utterance']

        doct_dict['date'] = date[:4]
        doct_dict['topic'] = topic

        for paragraph in paragraphs:
            doct_dict['sentences'].append(paragraph['form'])

        tag_bowl.append(doct_dict)

        data = pd.DataFrame(tag_bowl)
        
    return  data,middle_path_element


# In[15]:


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
    
    # 폴더에서 필터한 폴더를 반환
    fited_jason_forder = filter_jason_folder(folder_path)
    #print(f'제이슨 폴더: {fited_jason_forder}')

    # 분석할 내용을 담을 딕셔너리 생성 
    analysis_bowl = []

    # 폴더를 순회하면서 분석 내용 추출
    for jason_file in fited_jason_forder:

        # 파일 인코딩 체크
        file_encoding = detect_encoding(jason_file)
        
        with open(jason_file, 'r', encoding=file_encoding) as file:
            # 해당 데이터 
            data = json.load(file)
            # 파일 이름 생성 
            sentence_csv_path = create_file_name(folder_path,jason_file)
             # 데이터 프레임 추출 
            data,middle_path = extract_tag(data,path_elements) 
            # 데이터 리스트 추가
            analysis_bowl.append(data)

    # 데이터 병합
    concat_data = pd.concat(analysis_bowl)

    return concat_data,folder_path,middle_path         


# In[16]:


# Action: 파일 이름 생성 함수

def create_file_name(folder_path,jason_file):
    
    # 파일 이름만 추출
    file_name_with_extension = os.path.basename(jason_file)

    # 확장자 제거
    file_name, _ = os.path.splitext(file_name_with_extension)
    
    sentence_csv_path = os.path.join(folder_path, f"{file_name}.csv")

    return sentence_csv_path


# In[17]:


# Action: Jason 인코딩 감지 함수 

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:  # 파일을 바이너리 모드로 열기
        raw_data = file.read(10000)  # 파일의 첫 부분을 읽어 인코딩 감지 (전체 파일을 읽어도 되지만 메모리를 많이 사용할 수 있음)
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    return encoding


# In[18]:


def only_numbers(char):
    return char.isdigit()

def update_user_input(*args):
    user_input.delete(0, 'end')
    user_input.insert(0, tag_options[tag_variable.get()])


# In[19]:


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


# In[20]:


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


# In[21]:


def update_status_and_search():
    """
    로딩 함수 구현
    """
    try:
        # 로딩 상태 업데이트
        status_label.config(text="로딩 중...")
        root.update_idletasks()  # UI 업데이트 강제 실행

        # 기존 검색 함수 호출
        tag_part_sentence()

        # 작업 완료 후 상태 업데이트
        status_label.config(text="작업 완료")
        
    except:
        # 오류 발생 시 메시지 업데이트
        status_label.config(text="작업 실패:")
        print(f"오류 발생: {e}")


# In[ ]:


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
folder_button = ttk.Button(root, text="폴더 선택 및 분석", command=update_status_and_search)
folder_button.grid(row=2, column=2, sticky='ew', padx=10, pady=5)

# (2,2)
reset_button = ttk.Button(root, text="결과창 리셋", command=reset_table)
reset_button.grid(row=2, column=3, sticky='ew', padx=10, pady=5)

# (3,1)
# 상태 레이블
status_label = tk.Label(root, text="")
status_label.place(relx=0.5, rely=0.5, anchor='center')


root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(6, weight=1)
root.grid_rowconfigure(7, weight=1)

# 초기 캔버스 설정을 None으로 합니다. 'embed_figure' 함수 정의가 필요합니다.
canvas = None

root.mainloop()

