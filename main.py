import pandas as pd
import numpy as np
import streamlit as st

data = pd.read_csv('corrected_data.csv', sep=';', index_col=0)

likes_p_views = (pd.DataFrame({'month': pd.date_range(start=data.date.min(),
                                            end='2023-03-01',
                                            freq='1M')
                     .strftime("'%y %B")})
                                   .merge(data.groupby('month',
                                                       as_index=False)
                                          .agg({'likes': 'sum', 'views': 'sum'})
                                          .sort_values('month'),
                                      on='month',
                                      how='left'))

likes_p_views = likes_p_views.loc[np.flatnonzero(likes_p_views['views']).tolist()[0]:]
likes_p_views['likes_per_view'] = (likes_p_views['likes'] / likes_p_views['views']) * 100

likes_p_views = likes_p_views[['month', 'likes_per_view']]

st.bar_chart(data=likes_p_views, x='month', y='likes_per_view')

with open('text1.txt', "r", encoding='utf8') as f:
    text1 = f.read()
st.markdown(text1, unsafe_allow_html=True)

'''
Here is some text
'''
