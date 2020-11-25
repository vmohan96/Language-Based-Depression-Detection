# Detecting Depression From Language Using Machine Learning
Depression, often classified as major depressive disorder (MDD) is a psychological mood disorder characterized by excessive sadness, loss of interest in normal activities, as well as a number of additional symptoms that can directly and negatively affect a patient’s normal life. Depression is the most common of the mood disorders; as well as the leading cause of suicide worldwide. It affects tens of millions of people nationwide, and in the hundreds of millions across the world. 

While depression has been a widespread issue for a long time, there is no doubt that the recent COVID-19 pandemic has also led to a marked increase in rates of diagnosis. Studies have shown a 300% increase in symptom prevalence over 2019, and both depression and suicidal ideation among youth is at an all-time high. 

These are worrying statistics, and it is also an unfortunate fact that many cases of major depressive disorder, or depression due to other factors, are likely undiagnosed. Especially during a time when self-isolation is the norm, depression rates may be increasing even more than we are able to detect through voluntary screenings.

## 1. The Goal
What if we could detect warning signs of depression without a direct screening test or an in-person visit with a therapist? Specifically, what if we could design an automated algorithm to detect these warning signs for us?

These are precisely the questions that motivated this analysis and the creation of an NLP-based neural-network predictor. My primary goal for this investigation was to generate a model capable of taking in text from a user, who would talk about how they are feeling or doing, and generating an accurate prediction indicating the probability of that entry being indicative of depression.

## 2. Data
### 2.1 Training Set
In order to train a reasonably robust model capable of predicting depression, I knew I would need a particularly large text corpus. I decided to try and obtain a corpus with “depressive text,” as well as “neutral text,” so that a classifier would have a baseline/normalization to compare to. I decided to use Reddit as a source for these corpi, as this is a massive platform with a large and diverse user base, making it good for generalizability. I scraped 30,000 posts (titles and text) from r/depression, a support community for those struggling with this illness. This was an ideal dataset for my purposes, as each post consisted of the author writing a few sentences to a few paragraphs about how they were feeling or how they were contending with clinical depression.
Deciding on a neutral text corpus was slightly more tricky: I decided to scrape 15,000 posts from r/CasualConversation and r/happy, as these were both communities with posts similar in format to r/depression, but with an overall more neutral/slightly positive sentiment. For the purposes of the modeling process, I formatted the corpus to be two classes, either depressive or not depressive/neutral. 

### 2.2 DAIC-WOZ
Text-based depression detection is not an entirely new area of study; a few academic institutions have done studies on the subject. Frequently used in this field is a dataset known as the Distress Analysis Interview Corpus: this is a set of clinical interviews conducted by researchers with the intention of supporting diagnostic methods of depression as well as Post-Traumatic Stress Disorder (PTSD). 

A streamlined adaptation of this dataset, called the DAIC-WOZ, was created by the University of Southern California, and they were kind enough to share it with me. I primarily used this dataset for validation of my model, as well as some exploratory data analysis surrounding linguistic features of text from depressed participants.


## 3. bERT-BiLSTM Model
### 3.1 bERT
Previous studies in text-based mental health disorder detection have found a high degree of accuracy in classification when utilizing bERT word embeddings to represent text. Word embedding is a technique in which words and sentences are mapped onto higher dimensional vectors based on the contexts of their sentences. Word embeddings are particularly useful for NLP tasks that require more semantic or sentimental understandings of text. 

bERT is a particularly robust neural network transformer, utilizing both Masked Language Modeling as well as Next Sentence Prediction to bidirectionally improve understanding of multiple sentences, and specifically context. When using this for a specific task, the output layers are replaced to accommodate the dataset being analyzed, with the primary architecture being fine-tuned. 

In previous studies aiming to detect depression from text, bERT has been shown to capture semantic meaning most effectively, and best promote predictive strength. This was the primary motivator for using bERT to vectorize corpus text as the primary input to a neural network classifier.

In order to maximize efficiency, bERT embeddings were obtained from the corpus text using spaCy’s transformer library, and the text was vectorized using distilBERT, a more compact and efficient version of the larger library, that grants minimal losses in semantic meaning and accuracy.

### 3.2 Bi-LSTM Neural Network Classifier
Once the corpus text was converted to bERT word embeddings, it was passed into a Keras neural network classifier, with a Bidirectional LSTM (Long-Short-Term-Memory) layer. LSTM models use a recurrent network architecture, meaning they process data temporally and use information from previous time-steps to influence predictions at the current time-step. LSTM’s in particular are able to avoid a common problem with recurrent network architecture known as the vanishing gradient problem, which otherwise leads to the gradient of change from one time-step to the next to be increasingly small and the actual learning done by the model to be minimal. By accounting for long-term dependencies in sequential data, relevant data from many timesteps ago can be used to influence interpretation of data at the current timestep, which significantly mitigates vanishing gradients.

In the context of natural language processing, an LSTM network will process each word as its own input, and also allow previous words of a sentence to inform predictions from the current word. With the added benefit of aforementioned long-term-dependencies, words much earlier than the current word can still be processed if they are relevant, allowing the model to make more accurate predictions during the training process. When implementing a bidirectional framework, this process happens both forwards and backwards for each text input. 

This framework, combined with bERT’s sentence-level embeddings, allows for a robust neural network model that is able to capture meaning on the semantic and to an extent the sentimental level, in a way that is ideal for a task involving text describing feelings directly from the author.

## 4. Vader-Sentiments Neural Network
In the interest of specifically capturing sentiments of text in a way that bERT embeddings may not have done directly, a second model was built using sentiment scores for each data point in the corpus, based on Python’s vaderSentiment library. This model had one important strength over the bERT-BiLSTM in that it retains predictive power for shorter, simpler sentences, while the more complex model performs better with more verbose text inputs.

After testing with a section of the Reddit corpus as well as selected data points from the DAIC, the two models were folded together based on the class probabilities they outputted; specifically a weighted average that favored the BiLSTM model 70% to 30%. 


## 5. Results
The resultant combination model performed better than each one did individually. With a validation set from the Reddit corpus, the BiLSTM alone had an accuracy around 84%, the Vader model alone had about 77%, and the resultant combination model had about 86% accuracy. The increase is not a significant margin above the BiLSTM on its own, but a more important benefit was the model’s performance on exogenous data, which also seemed to be boosted. Samples from the DAIC, as well as a few from an online UK-based blog about depression called Time to Change, were used to test the model in real-time, and it performed well with both.

To increase availability and usability, this predictor was also saved and packaged as a tensorflow .hd formatted model, and I used it to build a Streamlit-based web application designed to function as a live text-based predictor of depression. This app also contains a 7-day “mood-tracker,” allowing the user to enter journal-style entries for 7 days, after which a plot is generated displaying the predicted progression of the user’s mood based on their text (using the same combined model). The application has been deployed Streamlit and can be accessed here:

[Detecting Depression from Text Using Machine Learning](https://share.streamlit.io/vmohan96/depression-detection-text-app/main.py)

In addition, I built a second, nearly identical application with the added functionality of speech to text recognition; this allows the user to speak directly into their microphone and dictate how they are feeling, and the application will the automatically recognize the text and run it through the predictive model, returning a probability of the text being correlated with depression. The speech to text functionality was achieved with the SpeechRecognition library from Python as well as Google Cloud’s Speech-to-Text API. This app was not able to be deployed to Streamlit at the time of creation due to incompatibility issues between Streamlit and PyAudio, but its source code can be seen within the Text-Depression-App-Master folder, in the ‘speech.py’ file.


## 6. Conclusions and Future Applications
### 6.1 Analyzing Language Data
After gaining an understanding of how each of the two models implemented performs, one significant takeaway from this analysis is that different models can work better/worse based on the length and complexity of text. For the most part, my initial bERT-BiLSTM model worked extremely well on the training dataset; this is a relatively robust language model working on relatively dense and complex language data. However, the vaderSentiment model, a much simpler architecture, worked better than the BiLSTM on shorter and simpler text. I view this as an important note; while I set out with this analysis expecting to build one model able to account for everything, it ended up being better to create two specialized models that each have their own strengths and are able to complement each other.


### 6.2 Possible Use Cases
One potential use case for this model and the more user-friendly application is as a complement for psychiatric therapy methods. Patients suffering from mood disorders like depression may visit a therapist once a week or less, and it is important to keep track of emotional state much more often than that. While mood tracking applications exist, the vast majority involve a direct input by the user indicating their mood on an arbitrary numerical scale. With an application like the one I built, patients could simply talk/record themselves on the platform in the same way they might talk to their therapist. The model would provide standardized predictions based on a standardized language model, which could potentially be more empirically useful for diagnostic methods.

On another note, a publicly available application that utilized this language model on the backend could allow for all its users to generate predictions based on their journal entries at any time, and also keep a mood tracker for themselves. With the proper permissions and consent (as well as a clear opt-out option for all users), this data could be anonymously collected and used to analyze depression on a more holistic scale, breaking it down by demographic or any other factors relevant to the analyzer, for potential public health applications.


## 7. References
- [The Distress Analysis Interview Corpus](https://dcapswoz.ict.usc.edu/)
- [The Distress Analysis Interview Corpus of human and computer interviews](http://www.lrec-conf.org/proceedings/lrec2014/pdf/508_Paper.pdf)
- [Text-based depression detection on sparse data](https://arxiv.org/pdf/1904.05154.pdf)