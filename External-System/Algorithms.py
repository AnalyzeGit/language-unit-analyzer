#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Option
import sys
import os
import json
import ijson
import csv
import pandas as pd
import chardet
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import numpy as np
import jpype
import jpype.imports
from jpype.types import *
import logging

# Natural Language processing 
from collections import Counter 
from konlpy.tag import Okt, Komoran, Hannanum, Kkma, Mecab

# Visualize 
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import font_manager, rc
from matplotlib.ticker import FuncFormatter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch, PathPatch
from io import BytesIO
from PyQt5.QtGui import QPixmap
import plotly.graph_objects as go
from PyQt5.QtCore import QUrl
import tempfile


# In[ ]:


import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import Qt  # Qt 모듈 임포트

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("signlab.ui")
form_class = uic.loadUiType(form)[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self) 

        # 저장 폴더,제외단어  및 켄버스 기본 값 설정 
        self.ste_folder_path = False
        self.canvas = None
        self.contents = None

        # morpheme ComboBox type 선택
        self.corp_type_cmb.currentIndexChanged.connect(self.select_and_set_corp_type)
        self.con_corp_type_cmb.currentIndexChanged.connect(self.select_and_set_corp_type)

        # morph분석버튼 연결
        self.anlys_btn.clicked.connect(self.select_final_language_analyzer)
        self.con_anlys_btn.clicked.connect(self.select_final_language_analyzer)
        
        # 폴더 버튼 연결
        self.bg_btn.clicked.connect(self.select_folder)
        self.con_bg_btn.clicked.connect(self.select_folder)

        # 저장위치 버튼 연결
        self.ste_btn.clicked.connect(self.select_storage_location)
        self.con_ste_btn.clicked.connect(self.select_storage_location)

        # 제외단어 버튼 연결
        self.fe_reg_btn.clicked.connect(self.upload_exclusion_word_file)

        # 지우기 버튼 연결
        self.rst_btn.clicked.connect(self.reset_canvas)
        self.con_rst_btn.clicked.connect(self.reset_canvas)

        # 형태소 분석기 초기화
        self.word_counter,self.morpheme_analyzers = self.initialize_morpheme_analyzers()

        # 폰트 이름 설정
        self.font_name = self.configure_font()



    def upload_exclusion_word_file(self):
        options = QFileDialog.Options()
        self.fe_reg_file_path,_ =  QFileDialog.getOpenFileName(self, "파일 선택", "", "All Files (*);;Text Files (*.txt)", options=options)
        # 파일에서 텍스트 읽기
        with open(self.fe_reg_file_path, 'r', encoding='utf-8') as file:
            self.content = file.read()
        # 텍스트 분리
        self.contents = self.content.split('\n')

    
    def select_storage_location(self):
        options = QFileDialog.Options()
        self.ste_folder_path = QFileDialog.getExistingDirectory(self, "폴더 선택", "", options=options)

        # 객체 확인
        sender = self.sender()

        # 객체가 concordance의 경우
        if sender.objectName() == 'con_ste_btn':
            self.con_ste_le.setText(self.ste_folder_path)
        else: # N-gram의 경우
            self.ste_le.setText(self.ste_folder_path)

    
    def select_folder(self):
        # 폴더 경로 설정
        options = QFileDialog.Options()
        self.folder_path = QFileDialog.getExistingDirectory(self, "폴더 선택", "", options=options) 

        
    def show_info_message(self, title, message):
        """
        메세지 박스 기능
        
        """
        QMessageBox.information(self, title, message, QMessageBox.Ok)

    
    def split_sentences(self, sentences, num_chunks):
        indices = np.linspace(0, len(sentences), num_chunks + 1, dtype=int)
        return [sentences[indices[i]:indices[i + 1]] for i in range(num_chunks)]

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
            "Are you sure you want to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            logging.info("Application is closing...")
            event.accept()
        else:
            event.ignore()

    def select_and_set_corp_type(self):
        """ 
        말뭉치 종류 선택 (Select Corpus Type)
        이 함수는 사용자가 콤보 박스에서 선택한 말뭉치 종류에 따라 
        해당 태그 옵션을 자동으로 라인 에디트 위젯에 채웁니다.
        """
        tag_options = {
        "신문 말뭉치": "document.paragraph.form",
        "일상 대화 말뭉치": "document.utterance.form",
        "직접 입력": ""}
        
        # 말뭉치 lineedit 자동 입력
        self.corp_tgt_le.clear() # 말뭉치 lineedit 초기화
        self.corp_tgt_le.insert(tag_options[self.corp_type_cmb.currentText()])

        self.con_corp_tgt_le.clear() # 말뭉치 lineedit 초기화
        self.con_corp_tgt_le.insert(tag_options[self.con_corp_type_cmb.currentText()])

    
    def initialize_morpheme_analyzers(self):

        # 1. 단어 통계를 위한 Counter 객체 생성
        word_counter = Counter()

        # 2. 형태소 분석기(Mecab 로드)

        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            # 현재 실행 파일의 디렉토리 경로
            base_dir = os.path.dirname(os.path.abspath(__file__))

        # Mecab 사전 경로 설정
        dicpath = os.path.join(base_dir, 'mecab-ko-dic')

        mecab = Mecab(dicpath=dicpath)

        # 3. 형태소 분석기 초기화
        morpheme_analyzers = {
            "선택 없음": None,
            "Okt": Okt(),
            "Komoran": Komoran(),
            "Hannanum": Hannanum(),
            "Kkma": Kkma(),
            'Mecab': mecab 
            }

        return word_counter, morpheme_analyzers
    
    
    def apply_generate_ngrams_preva(self,sentences,n):

        start_time = time.time()  # 함수 실행 전 시간 측정

        # 일꾼 개수 선택
        #num_workers = multiprocessing.cpu_count()
        num_workers = 4

        # 문장을 6개의 청크로 나누기
        chunks = self.split_sentences(sentences, num_workers)
    
        # 병렬 처리
        ngram_result = Parallel(n_jobs=num_workers)(delayed(self.process_chunk)(chunk,n) for chunk in tqdm(chunks,desc="Processing chunks"))

        # 결과를 리스트 컴프리헨션 방식으로 병합
        ngrams_comprehensions = [ngram for result in ngram_result for ngram in result]

        # 리스트를 데이터프레임으로 변환
        n_gram_dataset = pd.DataFrame(ngrams_comprehensions, columns=["n_grams"]).dropna().reset_index(drop=True)
        end_time = time.time()
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"엔 그램 함수_Execution time: {execution_time:.6f} seconds")

        return n_gram_dataset

    
    def apply_generate_ngrams(self,sentences,n):
        
        start_time = time.time()  # 함수 실행 전 시간 측정
    
        ngrams_comprehensions = [ngram for sentence in sentences for ngram in self.generate_ngrams(sentence, n)]
    
        n_gram_dataset = pd.DataFrame(ngrams_comprehensions,columns=['n_grams']).reset_index().dropna()
    
        end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"엔 그램 함수_Execution time: {execution_time:.6f} seconds")
    
        return n_gram_dataset

        
    def track_down_concordance_words(self):

        # 객체 확인
        sender = self.sender()

        # 객체가 콘코던스 페이지 경우
        if sender.objectName() == 'con_anlys_btn':
            concordance_words_get = self.srh_le.text()
            print(concordance_words_get)
            if concordance_words_get == '':
                self.show_info_message('Error',"검색어를 입력하세요")
                return 
            else:     
                concordance_words = concordance_words_get.split('|')
                concordance_words = [word.strip() for word in concordance_words]
                return  concordance_words
        # 엔그램 페이지 경우(콘코던스 기능 사용하지 않음)
        else:
            return 'None'

        
    def configure_font(self):

        #base_path = os.getcwd()

        if getattr(sys, 'frozen', False):  # 코드가 PyInstaller로 패키징된 경우
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(__file__)

        font_path = os.path.join(base_path, "fonts", "malgun.ttf")
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)

        return font_name
        
        
    def process_chunk(self,chunk, n):
        result = []
        for sentence in chunk:
            result.extend(generate_ngrams(sentence, n))
        return result
    
    
    def generate_ngrams(self,s, n):
        tokens = s.split()

        n_grams_dataset = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

        return n_grams_dataset
    
    def replace_strip(self,concordance_words_get):
        if concordance_words_get.strip() == '':
            return 'None'
        else:
            return concordance_words_get
        
        
    def execute_concordence_sentence_only_fuction(self):
        start_time = time.time()  # 함수 실행 전 시간 측정

        # 콘코던스 함수 실행
        concordance_words_get = self.track_down_concordance_words()
        # 문장 추출 함수 사용
        analyzed_folder= self.extract_materials_be_analyzed() 
        
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

            return  concordence_dict
        else:
            end_time = time.time()  # 함수 실행 후 시간 측정
            execution_time = end_time - start_time  # 실행 시간 계산
            print(f"콘코던스 함수_Execution time: {execution_time:.6f} seconds")

            return analyzed_folder
        
        
    def tag_part_sentence(self):

        start_time = time.time()  # 함수 실행 전 시간 측정

        # 분석 딕셔너리 
        sentence_dicts = self.execute_concordence_sentence_only_fuction() 

        # 제외 단어 추척 함수 사용
        exclude_words_entry = self.track_down_exclude_words()

        # n_grams 생성 
        n_gram = self.ng_le.text()

        # 표 표출 수 생성
        ex_num = int(self.ex_num_le.text())

        # 표 표출수 생성
        if ex_num!=0:
            table_limit_count = ex_num

        # 형태소 분석기 생성
        morpheme_menu_get = self.morph_cmb.currentText()

        # 아웃풋 딕셔너리 생성
        all_sentences = {'original': [], 'analyzed': []}

        for i,(key,value) in enumerate(sentence_dicts.items()):

            # 데이터 프레임 문장 추출
            sentences = value['Sentence']

            total = len(sentences)
        
            for i, sentence in enumerate(sentences):
                # 선택한 형태소 분석기로 문장을 형태소 분석합니다.
                morphemes = self.morpheme_analyzers[morpheme_menu_get].pos(sentence)

                # 제외 단어 목록에 포함되지 않은 형태소만 추가합니다.
                filtered_morphemes = [f"{word}/{tag}" for word, tag in morphemes if word not in exclude_words_entry]

                # 문장을 형태소 분석된 형태로 변환합니다.
                analyzed_sentence = ' '.join(filtered_morphemes)

                # 기존 문장 저장
                all_sentences['original'].append(sentence)

                # 분석된 문장 저장
                all_sentences['analyzed'].append(analyzed_sentence)
                
                
                progress_value = int((i + 1) / int(total) * 100)

                self.progress.setValue(progress_value)
            
                # 프로그레스바 설정
                #self.progress.valueChanged.connect(self.progress.value())

        # n-gram을 생성합니다.
        ngrams = self.apply_generate_ngrams(all_sentences['analyzed'], int(n_gram))

        # n_counts 생성
        ngrams_count = self.count_n_grams(ngrams)

        if table_limit_count!=0:
            ngrams_count = ngrams_count[:table_limit_count]

        else:
            pass

        pd.DataFrame(all_sentences).to_csv(f'{self.ste_folder_path}\\{self.file_name}.csv' ,index=False, encoding='utf-16')
        ngrams_count.to_csv(f'{self.ste_folder_path}\\{self.file_name}_ngram.csv', index=False, encoding='utf-16')

        # 네트워크 시각화 이미지 저장
        self.apply_network_chart(ngrams['n_grams'],all_sentences['analyzed'],self.ste_folder_path)

        # 시각화 함수
        self.plot_data(ngrams_count)

        # 워드 클라우드 이미지 저장
        self.create_word_cloud(ngrams_count,self.ste_folder_path)

        end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"태깅 함수_Execution time: {execution_time:.6f} seconds")

        return pd.DataFrame(all_sentences)
    
    
    def analyze_sentence(self,sentence, exclude_words_entry):
        
        morpheme_analyzer = mecab  # 함수 내부에서 형태소 분석기 객체를 생성합니다.
        morphemes = morpheme_analyzer.pos(sentence)
        filtered_morphemes = [f"{word}/{tag}" for word, tag in morphemes if word not in exclude_words_entry]
        
        return sentence, ' '.join(filtered_morphemes)

    
    def track_down_exclude_words(self):

        # 제외 단어 목록을 가져옵니다.
        exclude_word = [self.excl_le.text()]
        
        print(f'점검 단어: {self.contents}')
        
        if self.contents:
            print("제외단어 리스트 존재함")
            self.contents.append(exclude_word)
            return self.contents

        return exclude_word

    
    def separate_sentence_to_phrase(self):

        start_time = time.time()  # 함수 실행 전 시간 측정

        # 분석 딕셔너리 
        sentence_dicts = self.execute_concordence_sentence_only_fuction() 

        # 아웃풋 딕셔너리 생성
        all_sentences = {'file_name':[],'original': [], 'analyzed': []}

        total = len(sentence_dicts)
        
        for i, (key,value) in enumerate(sentence_dicts.items()):

            # 데이터 프레임 문장 추출
            sentences = value['Sentence']

            for sentence in sentences:
                # 선택한 형태소 분석기로 문장을 형태소 분석합니다.
                morphemes = sentence.split(' ')
                # 제외 단어 목록에 포함되지 않은 형태소만 추가합니다.
                filtered_morphemes = [f"{word}" for word in morphemes]

                # 문장을 형태소 분석된 형태로 변환합니다.
                analyzed_sentence = ' '.join(filtered_morphemes)

                # 기존 문장 저장
                all_sentences['original'].append(sentence)

                # 분석된 문장 저장
                all_sentences['analyzed'].append(analyzed_sentence)

                # 폴더 이름 저장
                all_sentences['file_name'].append(key)
            
            progress_value = int((i + 1) / total * 100)
            self.con_progress.setValue(progress_value)

        end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"태깅 함수_Execution time: {execution_time:.6f} seconds")

        return pd.DataFrame(all_sentences)    

    
    def split_sentence(self, sentence, exclude_words_entry):
        # 선택한 형태소 분석기로 문장을 형태소 분석합니다.
        morphemes = sentence.split(' ')

        filtered_morphemes = [f"{word}" for word in morphemes if word not in exclude_words_entry]

        return sentence, ' '.join(filtered_morphemes)


    def count_n_grams(self, words):
        # 단어 카운트
        word_count = Counter(words['n_grams'])

        word_count_data = pd.DataFrame(list(word_count.items()), columns=['Word','Frequency']).dropna()

        word_count_data = word_count_data.sort_values(by='Frequency', ascending=False)

        return word_count_data

        
    def filter_jason_folder(self):
        """ 필터 값에 맞는 JASON 데이터 추출하기 

        파라미터: 폴더 경로, 필터 키, 필터 값

        반환 값: 필터 값에 맞는 JASON을 추가한 폴더 
        """

        start_time = time.time()  # 함수 실행 전 시간 측정

        # 제이슨 폴더 생성
        jason_folder = []
        
        # 초기 값 설정
        cheked_file_path = None 
        for filename in os.listdir(self.folder_path):
            if (filename.endswith('.json')) | (filename.endswith('.JSON')):
                file_path = os.path.join(self.folder_path, filename)
                jason_folder.append(file_path)

        end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"폴더 필터링 함수_Execution time: {execution_time:.6f} seconds")

        return jason_folder        

    
    def tag_part_ft_sentence(self):
        start_time = time.time()  # 함수 실행 전 시간 측정
        
        # 분석 딕셔너리 
        sentences = self.filter_data()

        # 제외 단어 추척 함수 사용
        exclude_words_entry = self.track_down_exclude_words()

        # n_grams 생성 
        n_gram = self.ng_le.text()

        # 표 표출 수 생성
        ex_num = int(self.ex_num_le.text())

        # 표 표출수 생성
        if ex_num!=0:
            table_limit_count = ex_num

        # 형태소 분석기 생성
        morpheme_menu_get = self.morph_cmb.currentText()

        # 아웃풋 딕셔너리 생성
        all_sentences = {'original': [], 'analyzed': []}

        total = len(sentences)
        
        for i, sentence in enumerate(sentences):
            # 선택한 형태소 분석기로 문장을 형태소 분석합니다.
            morphemes = self.morpheme_analyzers[morpheme_menu_get].pos(sentence)

            # 제외 단어 목록에 포함되지 않은 형태소만 추가합니다.
            filtered_morphemes = [f"{word}/{tag}" for word, tag in morphemes if word not in exclude_words_entry]

            # 문장을 형태소 분석된 형태로 변환합니다.
            analyzed_sentence = ' '.join(filtered_morphemes)

            # 기존 문장 저장
            all_sentences['original'].append(sentence)

            # 분석된 문장 저장
            all_sentences['analyzed'].append(analyzed_sentence)
                
            progress_value = int((i + 1) / int(total) * 100)

            self.progress.setValue(progress_value)

        
        # n-gram을 생성합니다.
        ngrams = self.apply_generate_ngrams(all_sentences['analyzed'], int(n_gram))

        # n_counts 생성
        ngrams_count = self.count_n_grams(ngrams)

        if table_limit_count!=0:
            ngrams_count = ngrams_count[:table_limit_count]

        else:
            pass

        pd.DataFrame(all_sentences).to_csv(f'{self.ste_folder_path}\\{self.file_name}.csv' ,index=False, encoding='utf-16')
        ngrams_count.to_csv(f'{self.ste_folder_path}\\{self.file_name}_ngram.csv', index=False, encoding='utf-16')

        # 네트워크 시각화 이미지 저장
        self.apply_network_chart(ngrams['n_grams'],all_sentences['analyzed'],self.ste_folder_path)

        # 시각화 함수
        self.plot_data(ngrams_count)

        # 워드 클라우드 이미지 저장
        self.create_word_cloud(ngrams_count,self.ste_folder_path)

        end_time = time.time()  # 함수 실행 후 시간 측정
        execution_time = end_time - start_time  # 실행 시간 계산
        print(f"태깅 함수_Execution time: {execution_time:.6f} seconds")

        return pd.DataFrame(all_sentences)
        
    
    def filter_data(self):
        # 필터링 제이슨 폴더 추출 
        jason_folder = self.filter_jason_folder()
        dataframes_lt=[]
        
        for file_path in jason_folder:
            #파일 인코딩 체크
            file_encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=file_encoding) as file:
                data = json.load(file)
                
            dataframes = self.inspect_jason_filter_key(data)
            
            if dataframes is not None:
                dataframes_lt.append(dataframes)

        paragraphs = [paragraph for dataframe in dataframes_lt for paragraph in dataframe]
        if len(paragraphs)==0:
            self.show_info_message("Error",'필터 값에 맞는 데이터가 없습니다.')
            return 
        
        return paragraphs


    def inspect_jason_filter_key(self, jason_data):
        """
        필터 값과 일치하는 JASON 데이터 검사

        파라미터: 데이터, 필터 키, 필터 값, 데이터 경로

        반환 값: 필터 값에 맞는 JASON 데이터 경로
        """

        start_time = time.time()  # 함수 실행 전 시간 측정
        
        # 필터 키 분리
        filter_key_list = self.filter_key.split('.')
        paragraphs_lt =[]
        
        # 조건 실행
        try:
            jason_data = jason_data[filter_key_list[0]]
        except KeyError as e:
            print(f"KeyError : {e}")
            return None     
        
        try:
            for subval_ls in jason_data:
                paragraph = [subval_form['form'] for subval_form in subval_ls['paragraph']]
                subval_dt = subval_ls[filter_key_list[1]]
                json_dts = subval_dt[filter_key_list[2]]
                if json_dts==self.tag_name:
                    paragraphs_lt.append(paragraph)        
                                         
            end_time = time.time()  # 함수 실행 후 시간 측정
            execution_time = end_time - start_time  # 실행 시간 계산
            print(f"태그 필터링 함수_Execution time: {execution_time:.6f} seconds")
            words = [word for paragraphs in paragraphs_lt for word in paragraphs]
            return words

        except IndexError:
            return None
    
    def flatten_list(self, nested_list):
        flat_list = []
        for element in nested_list:
            if isinstance(element, list):  # 요소가 리스트인 경우, 재귀 호출
                flat_list.extend(self.flatten_list(element))
            else:
                flat_list.append(element)
        return flat_list                
                
                
    def extract_tag(self, data,path_elements):
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
                    result = [self.extract_tag(item, path_elements) for item in data]
                    # None 값을 제외한 결과만 필터링
                    return [item for item in result if item is not None]       
            # 현재 데이터가 딕셔너리이고 경로 요소가 키로 존재하는 경우
            elif isinstance(data, dict) and first_element in data:      
                # 다음 경로 요소로 재귀적으로 함수를 호출    
                return self.extract_tag(data[first_element], path_elements[1:])      
            elif (first_element==last_element):
                return data[first_element]           
        except KeyError as e:
                print(f"Path not found in the JSON structure: {e}")
        except Exception as e:
                print(f"An error occurred: {e}")           
                
                
                
    def extract_materials_be_analyzed(self):
        """ 폴더 필터링, 분석 태그 내용 컴바인 함수 구현

        반환 값: 테그 내용 리스트
        """
        # 태그 설정
        path_elements = self.corp_tgt_le.text().split('.')

        if path_elements==[""]:
            self.show_info_message("Error", "태그 값을 입력하세요")
            return   

        # 폴더에서 필터한 폴더를 반환
        fited_jason_forder = self.filter_jason_folder()
        # 분석할 내용을 담을 딕셔너리 생성 
        analysis_bowl = {}

        # 폴더를 순회하면서 분석 내용 추출
        for jason_file in fited_jason_forder:

            # 파일 인코딩 체크
            file_encoding = self.detect_encoding(jason_file)

            with open(jason_file, 'r', encoding=file_encoding) as file:
                # 해당 데이터 
                data = json.load(file)
                # 파일 이름 생성 
                sentence_csv_path = self.create_file_name(jason_file)
                 # 내용 추출 
                rows_bowl = self.extract_tag(data,path_elements) 
                # 리스트 정제         
                clean_list = self.flatten_list(rows_bowl)
                 # 데이터 프레임화
                clean_list = pd.DataFrame(clean_list,columns=['Sentence'])
                # 데이터 딕셔너리 추가
                analysis_bowl[sentence_csv_path] = clean_list

        return analysis_bowl                       
                
                
    def create_file_name(self,jason_file):

        # 파일 이름만 추출
        file_name_with_extension = os.path.basename(jason_file)
        return file_name_with_extension
    

    def detect_encoding(self, file_path):
        with open(file_path, 'rb') as file:  # 파일을 바이너리 모드로 열기
            raw_data = file.read(10000)  # 파일의 첫 부분을 읽어 인코딩 감지 (전체 파일을 읽어도 되지만 메모리를 많이 사용할 수 있음)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        return encoding
    
    
    def cut_window_sizes(self, sentences,key):
        # window_size 설정
        window_size_get = int(self.ctx_le.text())

        # 콘코던스 설정
        concordance_entry_get = self.track_down_concordance_words()

        if concordance_entry_get=='None':
            self.show_info_message("Information", "윈도우 사이즈를 사용하려면, 콘코던스 값을 입력해주세요")
            return 

        # 윈도우 사이즈 리스트 생성
        window_size_dict = {'파일명':[],'왼쪽 맥락':[], '검색어':[], '오른쪽 맥락':[]}

        # 문장 window_size 자르기
        for index,rows in sentences.iterrows():
            language_sentence = rows['original']
            # 문장을 공백 기준으로 단어로 분리
            words = language_sentence.split()
            file_name = rows['file_name']

            for concordance in concordance_entry_get:
                if concordance in language_sentence:
                    for i, word in enumerate(words):
                        if concordance in word:
                            target_index = i
                            break
                        else:
                            pass  # 조건을 만족하지 않는 경우 아무 동작도 하지 않음

                    start_index = max(0, target_index - window_size_get)
                    end_index = min(len(words), target_index + window_size_get + 1)
                    left_context = ' '.join(words[start_index:target_index])
                    right_context = ' '.join(words[target_index+1:end_index])
                    
                    if (len(words[start_index:target_index])>=window_size_get) and (len(words[target_index+1:end_index])>=window_size_get):
                        window_size_dict['파일명'].append(file_name)
                        window_size_dict['왼쪽 맥락'].append(left_context)
                        window_size_dict['검색어'].append(words[target_index])
                        window_size_dict['오른쪽 맥락'].append(right_context)
                else:
                    pass              

        window_size_dataset = pd.DataFrame(window_size_dict)
        group_window_size_dataset = window_size_dataset.dropna()
        #group_window_size_dataset =self.group_up(window_size_dataset)
        window_size_dataset.to_csv(f'{key}\\{self.file_name}_window_size.csv',index=False, encoding='utf-16')

        # 데이터 프레임을 HTML로 변환
        html_content = group_window_size_dataset.to_html()
        html_with_style =f"""
        <html>
        <head>
            <style>
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid black; padding: 10px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # HTML 콘텐츠를 임시 파일로 저장
        html_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        with open(html_file.name, 'w') as f:
            f.write(html_with_style)

        # Load the HTML file in QWebEngineView
        self.con_web_view.setUrl(QUrl.fromLocalFile(html_file.name))
        
        
    def group_up(self, data):
        #group_windows = data.groupby('windowsize_sentence').count()
        group_windows = data['windowsize_sentence'].value_counts().reset_index()
        return group_windows
        
    def choose_morph_ft_language_unit(self):
        return self.tag_part_ft_sentence()
      
        
    def choose_morph_language_unit(self):
        return self.tag_part_sentence()

        
    def choose_pharse_language_unit(self):
        return self.separate_sentence_to_phrase()
      
      
    def select_final_language_analyzer(self):

        # 저장 위치를 선택하지 않았을 경우
        if not self.ste_folder_path:
            self.show_info_message('Error','저장 위치를 입력하세요.')
            return
        
        start_time = time.time()  # 함수 실행 전 시간 측정

        # 객체 확인
        sender = self.sender()

        # 필터 키 설정
        self.filter_key = self.flt_le.text()
        print(f"필터키:{self.filter_key}")
        # 필터 값 설정
        self.tag_name = self.flt_val_le.text()
        print(f"필터값:{self.tag_name}")
        
        # 객체가 N-gram 객체 경우
        if (sender.objectName() == 'anlys_btn') and (self.filter_key == ''):
            print(f"필터키 없는 조건")
            self.file_name = self.fn_le_cv.text()   # 문서 저장 파일명
            self.png_name = self.fn_le_pg.text() # 그림 저장 파일명
            language_unit = self.choose_morph_language_unit()

            # 객체 파일명이 입력되지 않을 경우
            if (self.file_name =='') |(self.png_name ==''):
                self.show_info_message("Error", "파일명을 입력하세요")
                return 
            # 형태소 분석 완료 
            self.show_info_message("Information", "형태소 분석이 완료되었습니다! 결과가 저장되었습니다.")

        elif (sender.objectName() == 'anlys_btn') and (self.filter_key != None):
            print(f"필터키 있는 조건")
            self.file_name = self.fn_le_cv.text()   # 문서 저장 파일명
            self.png_name = self.fn_le_pg.text() # 그림 저장 파일명
            language_unit = self.choose_morph_ft_language_unit()

            # 객체 파일명이 입력되지 않을 경우
            if (self.file_name =='') |(self.png_name ==''):
                self.show_info_message("Error", "파일명을 입력하세요")
                return 
            # 형태소 분석 완료 
            self.show_info_message("Information", "형태소 분석이 완료되었습니다! 결과가 저장되었습니다.")
            
            
        # 객체가 Concordance의 경우    
        else:
            self.file_name = self.con_fn_le_cv.text() 
            language_unit = self.choose_pharse_language_unit()

            # 콘코던스 파일 명을 입력하지 않았을 경우우
            if (self.file_name ==''):
                self.show_info_message("Error", "파일명을 입력하세요")
                return

            # 데이터 셋 할당
            senetence_dataset = language_unit
            sentences = senetence_dataset[['file_name','original']]

            # 윈도우 사이즈 설정
            window_size_get = self.ctx_le.text()

            # 윈도우 사이즈가 존재할 경우
            if window_size_get != '' :
                # 윈도우 사이즈 함수 아용
                self.cut_window_sizes(sentences,self.ste_folder_path)
                # 완료 후 메세지 박스 출력
                self.show_info_message("Information", "형태소 분석이 완료되었습니다! 결과가 저장되었습니다.")

                end_time = time.time()  # 함수 실행 후 시간 측정
                execution_time = end_time - start_time  # 실행 시간 계산
                print(f"Execution time: {execution_time:.6f} seconds")
    
                return language_unit
            else:
                self.show_info_message("Error", "맥락을 입력하세요.")
                return 


            return language_unit

    def only_numbers(self, char):
        return char.isdigit()

    def reset_canvas(self):
        
        # QGraphicsScene에서 QGraphicsPixmapItem 제거
        self.web_view.setUrl(QUrl("about:blank"))

        self.con_web_view.setUrl(QUrl("about:blank"))\
        
        self.progress.setValue(0)
        self.con_progress.setValue(0)
        
        # 저장 위치, 파일명 리셋
        self.ste_le.clear()
        self.fn_le_cv.clear()
        self.fn_le_pg.clear()
        self.con_ste_le.clear()
        self.con_fn_le_cv.clear()
        self.srh_le.clear()
        self.ctx_le.clear()
        self.flt_le.clear()
        self.flt_val_le.clear()
        
    def plot_bar_graph(self,df, x_col, y_col, title="Morpho  Analysis Graph", x_title=None, y_title=None, rounded_corners=True, radius=10):

        colors = [
            'rgba(171, 99, 250, 0.8)',  # 보라색
            'rgba(93, 164, 214, 0.8)',  # 파란색
            'rgba(171, 99, 250, 0.6)',  # 더 연한 보라색
            'rgba(93, 164, 214, 0.6)',  # 더 연한 파란색
        ]
    
        fig = go.Figure()
        
        for i, row in df.iterrows():
            fig.add_trace(go.Bar(
                x=[row[x_col]],
                y=[row[y_col]],
                marker=dict(
                    color=colors[i % len(colors)],  # 막대에 대해 색상을 순환하여 설정
                    line=dict(color='rgba(58, 71, 80, 1.0)', width=1),
                    opacity=0.6,
                    pattern_shape="",
                    cornerradius=radius if rounded_corners else 0
                ),
                name=row[x_col],
                text=row[y_col],  # 텍스트를 y값으로 설정
                textposition='outside'  # 텍스트 위치를 막대 위로 설정
            ))
    
        fig.update_layout(
            title=title,
            xaxis_title=x_title if x_title else x_col,
            yaxis_title=y_title if y_title else y_col,
            bargap=0.2,
            plot_bgcolor='white',  # 배경을 흰색으로 설정
            paper_bgcolor='white',  # 배경을 흰색으로 설정
            width=900,  # 그래프의 너비 설정
            height=400,  # 그래프의 높이 설정
            margin=dict(l=50, r=50, t=50, b=50)  # 여백 설정
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')  # 가로 그리드 추가
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')  # 세로 그리드 추가
        
        fig.show
        
        return fig
    
    def plot_data(self, ngrams_count):

        try:
            # Get the number of items to display from the entry widget
            number_of_items = int(self.gh_num_le.text())
        except ValueError:  # In case of invalid input
            self.show_info_message("Error", "그래프 표출수를 입력하세요")
            return

        # Get the most common 'number_of_items' words
        common_words = ngrams_count[:number_of_items]

        # Separate the words and their counts
        words = list(common_words['Word'].values) 
        counts = list(common_words['Frequency'].values)

        # 데이터프레임 생성
        df = pd.DataFrame({
        'words': words,
        'counts': counts
        })
        
        # 함수 사용 예제
        self.fig = self.plot_bar_graph(df, x_col='words', y_col='counts', x_title='Words', y_title='Counts', radius=10)
    
        # Plotly figure to HTML
        html_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        self.fig.write_html(html_file.name)
        
        # Load the HTML file in QWebEngineView
        self.web_view.setUrl(QUrl.fromLocalFile(html_file.name))
      
        
    def create_word_cloud(self, data,file_path):

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
        wordcloud.to_file(f'{file_path}\\{self.png_name}.png')   
        
        
    def curved_edges(self, G, pos, ax, edge_colors, rad=0.2):
        for (u, v), color in zip(G.edges(), edge_colors):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            verts = [(x1, y1), ((x1 + x2) / 2, (y1 + y2) / 2 + rad), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)
            patch = PathPatch(path, facecolor='none', edgecolor=color, lw=2)
            ax.add_patch(patch)        
            
            
    def apply_network_chart(self, n_grams,sentences,file_path):

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

        self.visualize_network_chart(ngrams_adjMatrix,id2word,file_path)        

        
    def visualize_network_chart(self, ngrams_adjMatrix,labels,files_path):

        # numpy 배열을 그래프로 변환
        G = nx.from_numpy_array(ngrams_adjMatrix)

        # 엣지 색상과 투명도를 적절한 범위로 조정
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights)   # 엣지 최대 웨이트 저장
        edge_colors = [plt.cm.Reds (weight / max_weight) for weight in edge_weights]  # 색상 스케일 조정
        edge_alphas = [0.1 + (weight / max_weight) * 0.9 for weight in edge_weights]  # 투명도 스케일 조정

        # 노드 색상 설정 (붉은 계열)
        # 노드 색상 설정 (낮은 값은 보라색, 높은 값은 빨간색)
        node_color = [plt.cm.Reds (np.random.rand()) for _ in range(len(G.nodes()))]
        
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
            font_family=self.font_name,
            ax=ax)

        # 엣지 그리기
        for (u, v), color, alpha in zip(G.edges(), edge_colors,edge_alphas):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3, edge_color=[color], alpha=alpha)

        # 엣지 그리기 (곡선으로)
        self.curved_edges(G, pos, ax, edge_colors)

        # 네크워크 시각화 이미지 파일로 저장+
        plt.savefig(f'{files_path}\\{self.png_name}_net.png')

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    QMessageBox.critical(None, 'Error', f"An unhandled exception occurred:\n{error_msg}")
            
def main():
    # 전역 예외 처리기 설정
    sys.excepthook = handle_exception

    # 로그 설정: 파일에 기록
    logging.basicConfig(filename='app.log', level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    try:
        app = QApplication(sys.argv)
        myWindow = WindowClass()
        myWindow.show()
        sys.exit(app.exec_())
        
    except Exception as e:
        handle_exception(type(e), e, e.__traceback__)
        
if __name__ == "__main__":
    main()
   # app = QApplication(sys.argv)
    #myWindow = WindowClass()
    #myWindow.show()
    #sys.exit(app.exec_()) 


# In[ ]:




