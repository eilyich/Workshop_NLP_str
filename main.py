import numpy as np
import pandas as pd
import datetime
import streamlit as st
from streamlit_shap import st_shap
st.set_page_config(layout="wide")

import lightgbm

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import scipy
import string
import re
import nltk
from nltk.corpus import stopwords
import emoji
import spacy
import natasha
from wordcloud import WordCloud

emb = natasha.NewsEmbedding()
morph_tagger = natasha.NewsMorphTagger(emb)
morph_vocab = natasha.MorphVocab()
segmenter = natasha.Segmenter()
names_extr = natasha.NamesExtractor(morph_vocab)

RS = 1008 # random state
import shap
# shap.initjs()

# data = pd.read_csv('corrected_data.csv', sep=';', index_col=0)
#
# likes_p_views = (pd.DataFrame({'month': pd.date_range(start=data.date.min(),
#                                             end='2023-03-01',
#                                             freq='1M')
#                      .strftime("'%y %B")})
#                                    .merge(data.groupby('month',
#                                                        as_index=False)
#                                           .agg({'likes': 'sum', 'views': 'sum'})
#                                           .sort_values('month'),
#                                       on='month',
#                                       how='left'))
#
# likes_p_views = likes_p_views.loc[np.flatnonzero(likes_p_views['views']).tolist()[0]:]
# likes_p_views['likes_per_view'] = (likes_p_views['likes'] / likes_p_views['views']) * 100
#
# likes_p_views = likes_p_views[['month', 'likes_per_view']]
#
# st.bar_chart(data=likes_p_views, x='month', y='likes_per_view')
#



russian_alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдеёжзийклмнопрстуфхцчшщьыъэюя-'
bad_toponyms = ['владимир', '«ростов»', 'ростов', 'дон', 'таганрог', 'краснодар', 'москв', 'петербург', 'санкт']
special_chars = string.punctuation.replace('-', '') + '«»\t—…’/:'
custom_day_of_week_calendar = {
                            0: 'понедельник',
                            1: 'вторник',
                            2: 'среда',
                            3: 'четверг',
                            4: 'пятница',
                            5: 'суббота',
                            6: 'воскресенье'
                        }
post_source = 'vk' # CONST
all_texts = pd.read_csv('text_only.csv', sep=';', index_col=0)['text']
# all_texts.sample().tolist()[0] # random sample

with open('text1.txt', "r", encoding='utf8') as f:
    text1 = f.read()
st.markdown(text1, unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    attachments = st.number_input('Введите количество вложений', min_value=0, max_value=20, value=2, step=1)

with col3:
    source_group = st.selectbox(
            "Выберите группу для публикации",
            ('dspl_rostov', 'c52space', 'msiid', 'makaronka_space','rostov','artbazar_61',  'centr_step', 'd30space',
            'mayak.lektory',  'n_s_rnd',
           'sholokhovrostov'),
            # label_visibility=st.session_state.visibility,
            # disabled=st.session_state.disabled,
        )

text = st.text_area('Введите текст', height=300)

with st.expander("👇 Можно взять пример текста отсюда"):
    def display_random_text():
        rnd_text = all_texts.sample().tolist()[0]
        st.markdown(rnd_text)


    if st.button('Выбрать другой'):
        display_random_text()
    else:
        display_random_text()


@st.cache_resource
def some_operation(text=text):
    text = text.lower()
    is_pinned = 0
    len_text = len(text)
    text = re.sub(r"[.,:]", '', text)
    tags = re.findall(f"#[{russian_alphabet}]+", text)
    text = ' '.join([w for w in text.split(' ') if w not in tags])
    tags = " ".join([re.sub('#', '', word) for word in tags])

    has_link = re.findall("http", text)
    has_link = 1 if len(has_link) > 0 else 0

    exclamations_count = text.count('!')
    emoji_count = sum(1 for element in re.findall(":[A-z\\_\\-]+:", emoji.demojize(text)))
    for word in bad_toponyms:
        text = re.sub(f"{word}[\w]+ |{word}", '', text)

    nlp = spacy.load('ru_core_news_sm')
    doc = nlp(text)
    named_entities = [X.text for X in doc.ents]
    named_entities = ' '.join(named_entities)
    named_entities = re.sub(r"[^А-яё\-\s]+", '', named_entities)
    named_entities = re.sub(r" [А-яё]{0,2} |^[А-яё]{0,2} ", ' ', named_entities)
    named_entities = natasha.Doc(named_entities)

    named_entities.segment(segmenter)
    named_entities.tag_morph(morph_tagger)
    for token in named_entities.tokens:
        token.lemmatize(morph_vocab)
    named_entities = [i.lemma for i in named_entities.tokens]
    named_entities = list(set(named_entities))
    named_entities = ' '.join(named_entities)
    named_entities = re.sub(r"[ ]+", ' ', named_entities)
    named_entities = re.sub("рф", "россия", named_entities)
    named_entities = re.sub('российский', 'россия', named_entities)
    named_entities = re.sub(r' - |^- | -$', '', named_entities)

    tokenized = "".join([ch for ch in text if ch not in special_chars])

    tokenized = re.sub(f"[^А-я- ]+", '', tokenized)
    for el in ['\n', ' - ', ' – ', '- ', ' -']:
        tokenized = re.sub(el, ' ', tokenized)

    tokenized = re.sub(r" [А-яё]{0,2} |^[А-яё]{0,2} ", ' ', tokenized)
    tokenized = re.sub(r"[ ]+", ' ', tokenized)
    tokenized = re.sub(r"^[ ]|[ ]$", '', tokenized)

    actual_stopwords = stopwords.words('russian')
    actual_stopwords.extend(['это', 'год', 'весь', 'наш', 'свой', 'который'])

    natasha_string = natasha.Doc(tokenized)

    natasha_string.segment(segmenter)
    natasha_string.tag_morph(morph_tagger)
    for token in natasha_string.tokens:
        token.lemmatize(morph_vocab)

    pos = [w.pos for w in natasha_string.tokens]
    nouns = len([i for i in pos if i == 'NOUN'])
    verbs = len([i for i in pos if i == 'VERB'])
    adjectives = len([i for i in pos if i == 'ADJ'])

    tokenized = [w.lemma for w in natasha_string.tokens if w.lemma not in actual_stopwords]
    tokenized = ' '.join([i for i in tokenized])
    tokenized = re.sub(r' - |^- | -$', ' ', tokenized)

    tok_without_names = ' '.join([w for w in tokenized.split(' ') if w not in named_entities.split(' ')])
    tok_without_names_unique = ' '.join(list(set([w for w in tokenized.split(' ')
                                                  if w not in named_entities.split(' ')])))

    mean_word_len = round(np.mean([len(w) for w in tokenized.split(' ')]), 1)
    named_ent_count = len(named_entities.split())
    tags_count = len(tags.split())
    emoji_rate = emoji_count / len_text

    substr_don = '|'.join(['донскаягосударственнаяпубличнаябиблиотека',
                           'донскойгосударственнойпубличнойбиблиотеки',
                           'донскойпубличнойбиблиотеки',
                           'донскойпубличнойбиблиотеке',
                           'донскуюпубличнуюбиблитеку',
                           'донскойпубличнойбиблиотекой'])
    tags = re.sub(f"{substr_don}", 'донскаяпубличнаябиблиотека', tags)

    nouns = nouns / len(tokenized.split(' '))
    verbs = verbs / len(tokenized.split(' '))
    adjectives = adjectives / len(tokenized.split(' '))

    dayofweek = custom_day_of_week_calendar[datetime.datetime.today().weekday()]
    hour = datetime.datetime.now().hour

    features_num = [emoji_rate, exclamations_count,
                     nouns, verbs, adjectives, mean_word_len,
                     attachments, len_text,named_ent_count, tags_count]
    features_cat = [is_pinned, has_link, dayofweek, hour]
    features_text = [tok_without_names_unique, named_entities, tags]

    text_vectorizer = joblib.load('text_vectorizer.joblib')



    tags_vectorizer = joblib.load('tags_vectorizer.joblib')
    named_vectorizer = joblib.load('named_vectorizer.joblib')

    text_vec = text_vectorizer.transform([features_text[0]])
    named_vec = named_vectorizer.transform([features_text[1]])
    tags_vec = tags_vectorizer.transform([features_text[2]])


    ordinal_enc = joblib.load('ordinal_enc.joblib')
    ordinal_enc_views = joblib.load('ordinal_enc_views.joblib')

    features_cat_enc = ordinal_enc.transform([features_cat])

    X = np.hstack((text_vec.toarray(), tags_vec.toarray(), named_vec.toarray(),
                  [features_num],
                 features_cat_enc
                 ))

    date = datetime.datetime.today().strftime("%d %m %Y")  # CONST


    views_base_cols_cat = [date,
                          post_source,
                          source_group,dayofweek,
                          hour, is_pinned]
    views_base_cols_num = [attachments]

    X_views = scipy.hstack((ordinal_enc_views.transform([views_base_cols_cat]),
                            [views_base_cols_num]))
    booster = lightgbm.Booster(model_file='LGBM_Regressor.txt')

    conversion_prediction = booster.predict(X)


    booster_views = lightgbm.Booster(model_file='views_LGBM_Regressor.txt')



    likes_prediction = int(booster.predict(X)[0] * booster_views.predict(X_views)[0] / 100)



    # ----------------------------------------------------

    # with st.expander("Можно взять пример текста отсюда 👇"):
    #     def display_random_text():
    #         rnd_text = all_texts.sample().tolist()[0]
    #         st.markdown(rnd_text)
    #
    #
    #     if st.button('Выбрать другой'):
    #         display_random_text()
    #     else:
    #         display_random_text()

    # ----------------------------------------------------

    feature_names_all = pd.read_csv('feature_names.csv', sep=';', index_col=0)['features'].tolist()
    shap_explainer = joblib.load('shap_explainer.joblib')
    shap_values = shap_explainer.shap_values(X)
    # shap_graph = shap.force_plot(shap_explainer.expected_value, shap_values, X, feature_names=feature_names_all)
    return (conversion_prediction, likes_prediction,
        shap.force_plot(shap_explainer.expected_value, shap_values, X, feature_names=feature_names_all)
    )

def execute_operation():
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Execute the operation in chunks
    for i in range(10):
        # Update the status text
        status_text.text(f"Processing chunk {i+1} of 10")

        # Execute the operation for this chunk
        chunk_result = some_operation()

        # Update the progress bar
        progress_bar.progress((i + 1) / 10)

    # Remove the status text when the operation is complete
    status_text.empty()

    # Return the final result
    return chunk_result

# st_shap(shap_graph, height=200, width=1000)

if st.button("""✨ Угадать популярность"""):
    if text == '':
        st.markdown("""<p>  
  <big><span style="color: Coral;">Пожаллуйста введите текст :)</span></big>
</p>""", unsafe_allow_html=True)
    else:
        output = execute_operation()
    conversion = str(np.round(output[0], 1)[0])
    likes = str(output[1])
    if output[0] < 0.68:
        st.markdown(f"""Конверсия этого текста в лайки -(другими словами, согласно принятой нами метрике <b>популярность</b>)- составляет <i><font size="+4"><span style="color: Coral;">""" + conversion + """%""" + """</span></font></i>. Честно говоря, не очень-то он и популярен - его метрика хуже, чем у <b>85%</b> текстов, опубликованных в исследованных группах. Если бы этот текст был опубликован, он получил бы примерно <i><font size="+4"><span style="color: Coral;">""" + likes + """</span></font></i> лайков. \n \n Что повлияло на предсказание? Попробуем разобраться с помощью диаграммы ниже.""", unsafe_allow_html=True)
    elif output[0] >= 0.68 and output[0] < 2.2:
        st.markdown(f"""Конверсия этого текста в лайки или другими словами, <b>его популярность</b>)- составляет <i><font size="+4"><span style="color: Coral;">""" + conversion + """%""" + """</span></font></i>. Это неплохой результат - по результатам анализа он попадает в <b>45%</b>, скажем так, "добротных" тектстов. Если бы он был опубликован, то получил бы примерно <i><font size="+4"><span style="color: Coral;">""" + likes + """</span></font></i> лайков. \n \n Что повлияло на предсказание? Попробуем разобраться с помощью диаграммы ниже.""", unsafe_allow_html=True)
    elif output[0] >= 2.2 and output[0] < 3.8:
        st.markdown(f"""Конверсия этого текста в лайки или проще говоря, <b>его популярность</b>)- составляет <i><font size="+4"><span style="color: Coral;">""" + conversion + """%""" + """</span></font></i>. Что-ж, можно считать, это <>хороший</b> текст. По результатам анализа он в числе <b>25%</b> качественных постов, собравших немало лайков. Если бы этот текст был опубликован, то получил бы примерно <i><font size="+4"><span style="color: Coral;">""" + likes + """</span></font></i> лайков. \n \n Что повлияло на предсказание? Попробуем разобраться с помощью диаграммы ниже.""", unsafe_allow_html=True)
    else:
        st.markdown(f"""Конверсия этого текста в лайки другими словами, <b>его популярность</b>) или даже <font size="+4">популярность</font> - составляет <i><font size="+6"><span style="color: Coral;">""" + conversion + """%""" + """</span></font></i> 🔥. Это очень хороший текст, лучше только у классиков. Он в числе <b>15%</b> самых качественных постов, снискавших поддержку читателей. Если бы этот текст был опубликован, то получил бы примерно <i><font size="+4"><span style="color: Coral;">""" + likes + """</span></font></i> лайков. \n \n Что повлияло на предсказание? Попробуем разобраться с помощью диаграммы ниже.""", unsafe_allow_html=True)

    st_shap(output[2], height=200, width=1000)