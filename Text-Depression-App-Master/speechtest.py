import speech_recognition as sr
import streamlit as st
import numpy as np
from combinefunction import combine_predict
import plotly.express as px
import pandas as pd


st.title('Detecting Depression From Text (Voice Dictation)')

st.markdown("""Depression is the most common of mood disorders, and its effects on those who suffer from it can be severe. However, when we break down depression by the symptoms that characterize it, early detection begins to seem increasingly possible.

The application below implements a natural language processing model to classify spoken language; using machine learning to return a probability of input being indicative of depression. For more information on depression, feel free to visit the [National Institute for Mental Health’s website](https://www.nimh.nih.gov/health/topics/depression/index.shtml).
""")
st.subheader('Click the button below to begin recording, then dictate a journal-style entry about how you are feeling.')
st.markdown('You can be as verbose/brief as you would like, and include as many/few details as you would like. Once you dictate your entry, the model will automatically return a prediction indicating the probability of being indicative of depression.')
st.markdown('**Disclaimer: This application is NOT a verified diagnostic tool, and results should not be interpreted as official diagnoses.**')

warning_list = ['die', 'suicide', 'suicidal', 'death', 'dead', 'kill', 'end it', 'end my life', 'cut', 'hurt myself']
low_level = "It looks like you are doing well, we hope that's accurate! If not, we apologize and still encourage you to check out [SAMHSA's National Helpline.](https://www.samhsa.gov/find-help/national-helpline)."
mid_level = "Consider trying out a mental health tracker, such as [Sanvello](https://www.sanvello.com/) or [Ginger](https://www.ginger.io/). You can also read and share stories about mental health on platforms like [The Mighty](https://themighty.com/topic/depression/)."
high_level = "It looks like you might not being feeling well today. Remember that it's ok not to be ok! Don't be afraid to reach out to [SAMHSA's National Helpline.](https://www.samhsa.gov/find-help/national-helpline) Also check out free personal mental health resources like [Sanvello](https://www.sanvello.com/) or [Ginger](https://www.ginger.io/)."

def receive_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)

    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        output = r.recognize_google(audio)
        #print(f'Model Prediction for Depression Probability: {combine_predict(speech_output)}')
    except sr.UnknownValueError:
        output = "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        output = ("Could not request results from Google Speech Recognition service; {0}".format(e))

    return output

out = ''
if st.button('Click here to speak'):
    st.text('Say something!')
    out = receive_audio()

if out:
    st.markdown(f'**YOU SAID: \"{out}\"**')

if out:
    #K.set_session(session)

    prediction = combine_predict(out)

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
        if i in out:
            st.markdown('Your input may contain references to self-harm or suicide. If you are considering self-harm or suicide, please reach out to the [Crisis Text Line](https://www.crisistextline.org/topics/self-harm/#what-is-self-harm-1) or call the suicide hotline number at [800-273-8255](Tel:800-273-8255).')

