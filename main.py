import numpy as np
import pandas as pd
import datetime
import streamlit as st
from streamlit_shap import st_shap
st.set_page_config(layout="wide", page_title="Предсказание популярности публикаций", page_icon=":book:")
# st.set_page_config(page_title="Предсказание популярности публикаций", page_icon=":book:", layout="wide", theme="light")

import lightgbm
import shap

import joblib
import scipy
import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import emoji
import spacy
import natasha

emb = natasha.NewsEmbedding()
morph_tagger = natasha.NewsMorphTagger(emb)
morph_vocab = natasha.MorphVocab()
segmenter = natasha.Segmenter()
names_extr = natasha.NamesExtractor(morph_vocab)

RS = 1008  # random state

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
all_targets = pd.read_csv('target.csv', sep=';', index_col=0)

st.markdown("""<font size="+8">Анализ популярности публикаций</font>   
Проект преследует две взаимосвязанные цели:  
- проанализировать причины популярности/непопулярности тех или иных публикаций.  
- разработать модель машинного обучения, которая будет способна предсказывать популярность постов.  
   
Сам по себе термин "популярность" - достаточно общий - в вводной части исследования употребляется не случайно. Какой именно показатель выбран в качестве метрики популярности указано ниже.      
  
<img title="всё многообразие текста" alt="" src="https://raw.githubusercontent.com/eilyich/Workshop_NLP_str/master/wordclowd_usage.png" width="900">   


На иллюстрации наиболее часто встречающиеся слова: чем слово больше - тем чаще оно упоминается в публикациях. Здесь они, как бы "разобраны" на отдельные детали, из которых сооружена предиктивная модель. [Опробовать её можно ниже прямо сейчас](#0). 

----
<font size="+6">Немного об исходных данных</font>   
Почти все данные с платформы <span style="color: LightCoral;"><b>Телеграм</b></span> в тексе сообщения содержат ошибку. Кроме того, единственный в наборе данных признак, который можно использовать как меру популярности - ответы на публикации - также отсутствует в подавляющем большинстве случаев. С учётом этого, анализ постов в Телеграмме вынесем **за рамки** исследования. При необходимости работы именно с данными из этого мессенджера, наилучшим решением будет новый сбор данных и отдельное исследование, которое можно провести на базе этого.   
""", unsafe_allow_html=True)
st.markdown("""

Данные из <span style="color: LightCoral;"><b>VK</b></span> гораздо более полные, присутствует исчерпывающее количество признаков, частично отсутствуют данные только в поле "текст" (это приемлемо в данном исследовании). Также, имеется сразу несколько признаков, которые могут быть избраны как целевые для вычисления метрики популярности.  

<font size="+6">Выбор целевого признака</font>   
Среди признаков, которые могут быть выбраны как целевые - лайки, репосты, комментарии и просмотры. Однако, выбор любой из них повлёк бы дальнейшие проблемы при интерпретации результатов исследования.   


- `комментарии` - даже в пиковое время количество комментариев в месяц не превышает 200. Для многих постов комментарии отсутствуют, но это не делает эти публикации плохими. Кроме того, комментарии могут быть негативными. При условии, если бы комментарии сопровождали каждый или почти каждый пост, популярность публикаций можно было бы оценивать, проводя сантиментный анализ комментариев. Однако, в данных условиях, а так же с учётом, что комментарии не выгружены для изучения - это не приоритетная задача.    


- `репосты` - неактуальная метрика. Репосты, безусловно, имеют важное значение для продвижения, однако причины по которым их делают пользователи, далеки от исследования популярности. Например, репост может сделать сотрудничающая группа или организация - исключительно в партнёрских целях.   


- `лайки` - измерение количества лайков, как мера популярности публикации даёт эффект наиболее близкий к желаемому. Однако, показатели в 100 лайков в группе с 1 тыс. пользователей и в группе со 100 тыс. пользователей имеют принципиальное различие.  


- `просмотры` - показатель, рост или снижение которого, не связаны напрямую с качеством публикации. Например, просмотров может быть больше, если пост размещён утром, так как впереди целый день пользовательской активности. И, куда более важно, что просмотры - это не итоговый показатель, их количество может увеличиваться с течением длительного времени, даже спустя год или два.

<font size="+4">Синтетическая метрика</font>   
Правильным решением, в данной ситуации, будет измерение конверсии просмотров в лайки.       
<img title="синтетический целевой признак (распределение)" alt="" src="https://raw.githubusercontent.com/eilyich/Workshop_NLP_str/master/conv.png" width="900">  

Итак, в среднем пост лайкают 2.3% из тех, кто его видел. Выбросы свыше 7.8% конечно имеют место быть, однако на графике выше они лишь для наглядности - из анализа такие посты были исключены, чтобы "не сбивать" модель.
  
<font size="+6">Модель</font>    
Для предсказания конверсии просмотров в лайки была подготовлена регрессионная модель на основе библиотеки <span style="color: LightCoral;"><b>LightGBM</b></span>. Она была обучена на 8284 тренировочных записях, и проверена на 2071 тестовой. Вот результат этой проверки.    
Метрики, полученные путём предсказания на тестовой выборке:   
<span style="color: Coral;">RMSE (Корень из среднеквадратичной ошибки) = <font size="+5">1.11</font> при стандартном отклонении 1.56</span>   
- <font size="+1">(RMSE меньше стандартного отклонения - значит предсказывает лучше, чем если бы мы взяли среднее значения конверсии для всех записей. Чем меньше RMSE - тем лучше.)</font>     

<span style="color: Coral;">R² (Коэффициент детерминации) = <font size="+5">0.5</font></span>   
- <font size="+1">(R² показывает насколько модель в целом хороша. Если метрика выше ноля - значит с этой моделью можно работать. Чем ближе значение к 1.0 - тем модель лучше. Метрика меньше нуля означает что что-то не так с моделью или данными.)</font>     

А вот как выглядит отклонение предсказаний модели от истинных значений (да, отклонения это нормально - ничто не идеально):    
<img title="синтетический целевой признак (распределение)" alt="" src="https://raw.githubusercontent.com/eilyich/Workshop_NLP_str/master/residual.png" width="600">   

Как видно отклонения, в целом, в пределах процента от истинной конверсии.   
А теперь, почему бы не написать текст самостоятельно (или воспользоваться одним из имеющихся)?     

---   
<a id="0"></a>      
<font size="+6">Применение модели</font>   

В этой "песочнице" можно попробовать написать свой текст, или позаимствовать что-нибудь "из классики", а затем оценить - в сердцах какого процента пользователей он нашёл бы отклик, и сколько лайков ему бы поставили. Прикладывать картинки не нужно - достаточно вообразить их, и установить число вложений (по умолчанию их два). Также можно поменять группу, в которой выходит публикация - это не повлияет на предсказание доли конверсии, но окажет эффект на предсказание лайков в абсолютных единицах.  

""", unsafe_allow_html=True)

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
        rndm = all_texts.sample()
        rnd_text = rndm.tolist()[0]
        rnd_convertion = str(np.round(all_targets.loc[rndm.index[0]][0], 1))
        st.markdown(rnd_text)
        st.markdown("""   
        ---
        
        
        ```(В реальности этот текст лайкнули """ + rnd_convertion + """% пользователей)```""", unsafe_allow_html=True)
        
        st.markdown("""<font size="+1">Примечание: даже при идеальном совпадении анализа модели с реальностью, в итоговом предсказании будет отличие - например из-за времени суток и дня недели, в котороый вы осуществляете проверку. Кроме того, стоит иметь в виду, что модели легче предсказывать популярность длинных текстов. Анализ короткич, из одного-двух слов, напротив не будет релевантным. </font>""", unsafe_allow_html=True)


    if st.button('Выбрать другой'):
        display_random_text()
    else:
        display_random_text()


@st.cache_resource
def blackbox(text=text):
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
                 features_cat_enc,
                 ))

    X[0][-1] = hour

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

    feature_names_all = pd.read_csv('feature_names.csv', sep=';', index_col=0)['features'].tolist()
    shap_explainer = joblib.load('shap_explainer.joblib')
    shap_values = shap_explainer.shap_values(X)
    # shap_graph = shap.force_plot(shap_explainer.expected_value, shap_values, X, feature_names=feature_names_all)
    return (conversion_prediction, likes_prediction,
        shap.force_plot(shap_explainer.expected_value, shap_values, X, feature_names=feature_names_all)
    )

def execute_operation(text):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Execute the operation in chunks
    for i in range(10):
        # Update the status text
        status_text.text(f"Processing chunk {i+1} of 10")

        # Execute the operation for this chunk
        chunk_result = blackbox(text)
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
  <big><span style="color: Coral;">Пожалуйста введите текст :)</span></big>
</p>""", unsafe_allow_html=True)
    else:
        output = execute_operation(text)
    conversion = str(np.round(output[0], 1)[0])
    likes = str(output[1])
    if output[0] < 0.68:
        st.markdown(f"""Конверсия этого текста в лайки -(другими словами, согласно принятой нами метрике популярность- составляет <i><font size="+4"><span style="color: Coral;">""" + conversion + """%""" + """</span></font></i>. Честно говоря, не очень-то он и популярен - его метрика хуже, чем у <b>85%</b> текстов, опубликованных в исследованных группах. Если бы этот текст был опубликован, он получил бы примерно <i><font size="+4"><span style="color: Coral;">""" + likes + """</span></font></i> лайков. \n \n Что повлияло на предсказание? Попробуем разобраться с помощью диаграммы ниже.""", unsafe_allow_html=True)
    elif output[0] >= 0.68 and output[0] < 2.2:
        st.markdown(f"""Конверсия этого текста в лайки или другими словами, его популярность- составляет <i><font size="+4"><span style="color: Coral;">""" + conversion + """%""" + """</span></font></i>. Это неплохой результат - по результатам анализа он попадает в <b>45%</b>, скажем так, "добротных" текстов. Если бы он был опубликован, то получил бы примерно <i><font size="+4"><span style="color: Coral;">""" + likes + """</span></font></i> лайков. \n \n Что повлияло на предсказание? Попробуем разобраться с помощью диаграммы ниже.""", unsafe_allow_html=True)
    elif output[0] >= 2.2 and output[0] < 3.8:
        st.markdown(f"""Конверсия этого текста в лайки или проще говоря, его популярность- составляет <i><font size="+4"><span style="color: Coral;">""" + conversion + """%""" + """</span></font></i>. Что-ж, можно считать, это <b>хороший</b> текст. По результатам анализа он в числе <b>25%</b> качественных постов, собравших немало лайков. Если бы этот текст был опубликован, то получил бы примерно <i><font size="+4"><span style="color: Coral;">""" + likes + """</span></font></i> лайков. \n \n Что повлияло на предсказание? Попробуем разобраться с помощью диаграммы ниже.""", unsafe_allow_html=True)
    else:
        st.markdown(f"""Конверсия этого текста в лайки другими словами, его популярность или даже <font size="+4">популярность</font> - составляет <i><font size="+6"><span style="color: Coral;">""" + conversion + """%""" + """</span></font></i> 🔥. Это очень хороший текст, лучше только у классиков. Он в числе <b>15%</b> самых качественных постов, снискавших поддержку читателей. Если бы этот текст был опубликован, то получил бы примерно <i><font size="+4"><span style="color: Coral;">""" + likes + """</span></font></i> лайков. \n \n Что повлияло на предсказание? Попробуем разобраться с помощью диаграммы ниже.""", unsafe_allow_html=True)

    st_shap(output[2], height=200, width=1000)
    st.markdown("""
    <span style="color: #0a6afa;">Синие</span> стрелки давят на метрику (в нашем случае потенциальную конверсию в лайки) слева, стремясь уменьшить её. <span style="color: #f23d61;">Красные</span> стрелки толкают метрику наверх, к популярности. Чем шире стрелка, тем больше влияние этого признака на итоговый результат. Ну а сама метрика зажата посередине - это число, вероятней всего дробное, выделенное жирным. В зависимости от текста, вы можете увидеть в числе влиятельных признаков слова - которых не писали. Это нормально, например признак "музей = 0", давящий на метрику справа, означает что отсутствие упоминания каких-либо музеев немного снижает метрику. """, unsafe_allow_html=True)


st.markdown("""<font size="+6">Подробнее о признаках</font>    
Текст, в данном анализе, является, по сути, главным признаком. Чем "весомее" слово, тем сильнее оно смещает оценку вероятной популярности в ту или иную сторону.  

<img title="сила слов" alt="" src="https://raw.githubusercontent.com/eilyich/Workshop_NLP_str/master/wordclowd_all.png" width="900">    

Это облако слов отличается от того, что представлено в начале. На первом плане другие слова - теперь самые большие это те, что имеют наибольший вес в модели. Музеи, художники, выставки, библиотеки - учитывая специфику изученных публикаций, нет ничего удивительно в том, что эти сущности вызывают отклик у читателей. Анализ важности текстовых признаков в большей степени даёт **понимание тем**, которые находят отзыв у подписчиков, нежели **способа подачи** информации.  


<img title="важность текстовых признаков" alt="" src="https://raw.githubusercontent.com/eilyich/Workshop_NLP_str/master/word_feature_importances.png" width="900">   

""", unsafe_allow_html=True)

st.markdown("""
        Некоторые сущности (слова), как то "стать", "день", "друг", "работа" существенно влияют на предсказание модели, однако интерпретировать их без контекста не представляется возможным.   
Такие сущности как "выставка", "музей", "фонд", "художник", "библиотека" дают понимание, что интересует публику.   

Однако не все эти сущности в тексте обязательно повышают популярность публикаций. Некоторые, влияют на неё негативно.""", unsafe_allow_html=True)

st.markdown("""
<div class="pull-left">
    <img title="shap values for text" alt="" src="https://raw.githubusercontent.com/eilyich/Workshop_NLP_str/master/shap_text_summary2.png" width="600">
    </div>""", unsafe_allow_html=True)


st.markdown("""
▲ Как именно текстовые признаки влияют на целевой иллюстрирует график.  
<span style="color: red;">Красным</span> выделены высокие значения признака (в случае с текстом это $1$ - значит слово есть в публикации). <span style="color: blue;">Синим</span> выделены низкие значения признака ($0$ - слова нет в тексте).
Значения в левой части графика - то, что понижает предсказание (меньше лайков), в правой части графика - то, что повышает.

Видно, что когда речь о музеях, фондах, выставках, пользователи действительно охотней отмечают эти публикации. 
Положительный интерес вызывают также упоминания библиотеки, художников, экскурсий, открытий (вероятно, каких-либо мероприятий).  

В то же время, такие сущности как "билет", "вход", "вопрос", "новый" - влияют негативно на популярность. Важно отметить: не стоит воспринимать такие понятия как отпугивающие - они лишь вносят вклад в общее восприятие и реакцию пользователей.  

Окончательная интерпретация значимости этих сущностей - в ведении авторов и редакторов публикаций.   

----   


Именованные сущности также влияют не предсказание модели, однако интерпретировать их не представляется возможным, так как в большинстве случаев это имена людей и топонимы - популярность постов с их упоминанием зависит от контекста.

 """, unsafe_allow_html=True)

st.markdown("""НЕтекстовые признаки в большей степени демонстрируют то **как** а **не о чём** должен быть написан текст, чтобы понравиться читателям.    



<img title="важность прочих признаков" alt="" src="https://raw.githubusercontent.com/eilyich/Workshop_NLP_str/master/other_feature_importances.png" width="900">   

""", unsafe_allow_html=True)

st.markdown("""График ниже также помогает детальнее разобраться в контексте влияния НЕтекстовых признаков.   """, unsafe_allow_html=True)
st.markdown("""
    <div class="pull-left">
        <img title="shap values for other features" alt="" src="https://raw.githubusercontent.com/eilyich/Workshop_NLP_str/master/shap_other_summary2.png" width="600">
        </div>""", unsafe_allow_html=True)



st.markdown("""
            Хорошо видно, что короткие, лаконичные тексты пользуются большей популярностью среди пользователей.    


Также положительно на количестве лайков сказывается отсутствие или минимальное количество эмодзи (вероятно, это малоприемлемо именно в культурной повестке).   

Популярность постов повышает наличие тэгов, однако важно, чтобы их не было слишком много. На графике слева можно наблюдать, что зона с ярко-красными точками (большое количество тэгов) находится примерно посередине диаграммы, в то время как в правой части "зоны повышения" - более тёмная область (небольшое количество тэгов).   

Вложения также, потенциально, повышают популярность публикаций. Впрочем, большинство наблюдений лежит не в зоне роста или убывания популярности, а посередине, из чего можно сделать вывод, что влияние этого параметра не критично. Кроме того, без сомнения, популярность публикации в значительной степени зависит от того какое именно вложение/вложения есть в посте. Именно по этой причине в датаесете для анализа были оставлены публикации без текста - фотография, или альбом фотографий, к примеру на художественную тематику, могут привлечь гораздо больше внимания.  
 
Что касается самой манеры написания текстов: невозможно однозначно интерпретировать влияние средней длинны слов на популярность. Бóльшая доля существительных положительно влияет на количество лайков - можно это интерпретировать, как склонность пользователей к получению конкретики. Сделать такого же однозначного вывода о долях глаголов и прилагательных в тексте невозможно: влияние есть, но оно незначительное.  

     """, unsafe_allow_html=True)

st.markdown("""
-----    

Для достижения повышения конверсии просмотров в лайки также можно ориентироваться на подходящее время публикации. Видно, что посты опубликованные в понедельник и в середине недели в среднем популярнее. Также больше поддержки получают тексты, опубликованные утром/в первой половине дня.   

<img title="зависимость успеха публикации от времени" alt="" src="https://raw.githubusercontent.com/eilyich/Workshop_NLP_str/master/time_dependences.png" width="900">

""", unsafe_allow_html=True)

st.markdown("""
<font size="+6">Выводы</font>    
В целом при формировании публикаций следует руководствоваться стандартными правилами написания текстов - публицистических, если речь об описании событий, или маркетинговых, если речь о продвижении. Писать "по сути" (выше доля существительных), сжато (в разумных пределах - меньше длинна текста, средняя длинна слов), не злоупотреблять эмодзи, для наиболее важных постов выбирать время публикации в первой половине дня.  
""", unsafe_allow_html=True)