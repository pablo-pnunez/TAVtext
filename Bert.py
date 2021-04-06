import tensorflow_hub as hub
import tensorflow_text

# Load BERT and the preprocessing model from TF Hub.
preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')
encoder = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')

# Use BERT on a batch of raw text inputs.
input = preprocess(['Batch of inputs', 'TF Hub makes BERT easy!', 'More text.'])
pooled_output = encoder(input)["pooled_output"]
print(pooled_output)
