import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.express as px
import pandas as pd
import numpy as np
from combinefunction import combine_predict
import plotly.express as px


st.title('Detecting Depression from Text Using Machine Learning')

st.markdown("""Depression is the most common of mood disorders, and its effects on those who suffer from it can be severe. However, when we break down depression by the symptoms that characterize it, early detection begins to seem increasingly possible.

The application below implements a natural language processing model to classify text passed in; using machine learning to return a probability of text being indicative of depression. For more information on depression, feel free to visit the [National Institute for Mental Health’s website](https://www.nimh.nih.gov/health/topics/depression/index.shtml).
""")


st.subheader('Please enter below a journal-style entry about how you are doing today/recently.')
st.markdown('You can be as verbose/brief as you would like, and include as many/few details as you would like. Once you enter your text, the model will return a prediction indicating the probability of being indicative of depression.')
st.markdown('**Disclaimer: This application is NOT a verified diagnostic tool, and results should not be interpreted as official diagnoses.**')

# Single Input Prediction
text_input = st.text_area('Answer here:')
warning_list = ['die', 'suicide', 'suicidal', 'death', 'dead', 'kill', 'end it', 'end my life', 'cut', 'hurt myself']
low_level = "It looks like you are doing well, we hope that's accurate! If not, we apologize and still encourage you to check out [SAMHSA's National Helpline.](https://www.samhsa.gov/find-help/national-helpline)."
mid_level = "Consider trying out a mental health tracker, such as [Sanvello](https://www.sanvello.com/) or [Ginger](https://www.ginger.io/). You can also read and share stories about mental health on platforms like [The Mighty](https://themighty.com/topic/depression/)."
high_level = "It looks like you might not being feeling well today. Remember that it's ok not to be ok! Don't be afraid to reach out to [SAMHSA's National Helpline.](https://www.samhsa.gov/find-help/national-helpline) Also check out free personal mental health resources like [Sanvello](https://www.sanvello.com/) or [Ginger](https://www.ginger.io/)."


if text_input:

    prediction = combine_predict(text_input)

    if prediction < 0.2:
        message = 'Your input is unlikely to correlate with depression.'
        rec = low_level
        color='#27B240'
    elif 0.2 <= prediction < 0.4:
        message = 'Your input is moderately unlikely to correlate with depression.'
        rec = mid_level
        color='#F0D637'
    elif 0.4 <= prediction < 0.6:
        message = 'Your input is moderately likely to correlate with depression.'
        rec = mid_level
        color = '#F08837'
    elif 0.6 <= prediction < 0.8:
        message = 'Your input is likely to correlate with depression.'
        rec = high_level
        color = '#EA4915'
    else:
        message = 'Your input is highly likely to correlate with depression.'
        rec = high_level
        color = '#970000'


    df = pd.DataFrame({'Probability of Being Depressive Text':prediction},index=['P'])
    fig = px.bar(df, 
                y=[0.8], 
                x="Probability of Being Depressive Text", 
                orientation='h',
                range_x=[0,1],
                range_y=[0,3],
                color_discrete_sequence =[f'{color}'],
                barmode='group')
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(title_text="Your Text's Probability of Correlating with Depression")
    fig.update_yaxes(title_text='')

    fig
    st.subheader(message)
    st.markdown(rec)

    for i in warning_list:
        if i in text_input:
            st.markdown('Your input may contain references to self-harm or suicide. If you are considering self-harm or suicide, please reach out to the [Crisis Text Line](https://www.crisistextline.org/topics/self-harm/#what-is-self-harm-1) or call the suicide hotline number at [800-273-8255](Tel:800-273-8255).')


st.text("")
st.text("")
st.text("")
st.text("")



# Mood Tracker
st.title('Mood Tracker')
st.subheader('Enter journal entries indicating how you have been feeling over 7 days')

day_1 = st.text_input('Answer Day 1:')
day_2 = st.text_input('Answer Day 2:')
day_3 = st.text_input('Answer Day 3:')
day_4 = st.text_input('Answer Day 4:')
day_5 = st.text_input('Answer Day 5:')
day_6 = st.text_input('Answer Day 6:')
day_7 = st.text_input('Answer Day 7:')

week = {1:day_1, 2:day_2, 3:day_3, 4:day_4, 5:day_5, 6:day_6, 7:day_7}

fig_week = px.scatter(
        range_x=[1,7], 
        range_y=[0,1],)

get_mood = st.button('Track my mood!')

if get_mood:
    xs = []
    ys = []
    for num,day in week.items():
        if day:
            mood_score = 1 - combine_predict(day)
            xs.append(num)
            ys.append(mood_score)
            
    fig_week.add_scatter(x=xs,y=ys,line=dict(color="#34D19F"))
    fig_week.update_layout(title='Your Predicted Mood This Week')
    fig_week.update_yaxes(title_text='Mood Score (1 is positive)')
    fig_week.update_xaxes(title_text='Day')

    fig_week