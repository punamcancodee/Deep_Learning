
##tensorflow >2.0
from tensorflow.keras.preprocessing.text import one_hot

### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

sent

voc_size=10000
onehot_repr=[one_hot(words,voc_size)for words in sent]
print(onehot_repr)

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

import numpy as np

sent_length=8
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

dim=10

model=Sequential()
model.add(Embedding(voc_size,10,input_length=sent_length))
model.compile('adam','mse')

model.summary()

print(model.predict(embedded_docs))

embedded_docs[0]

print(model.predict(embedded_docs)[0])