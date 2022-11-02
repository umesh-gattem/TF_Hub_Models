import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import sys

# paragraph  =

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

questions = [
    'Who built Taj Mahal?',
    'Who is 14th child of Mumtaz Mahal?',
    'When did Taj Mahal built?']
paragraph = '''The Taj Mahal was commissioned by Shah Jahan in 1631, to be built in the memory of his wife 
Mumtaz Mahal, who died on 17 June that year, while giving birth to their 14th child, Gauhara Begum.
Construction started in 1632, and the mausoleum was completed in 1648, while the surrounding buildings and 
garden were finished five years later. The imperial court documenting Shah Jahan's grief after the death of 
Mumtaz Mahal illustrates the love story held as the inspiration for the Taj Mahal.'''

for question in questions:
    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(paragraph)
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    # using `[1:]` will enforce an answer. `outputs[0][0][0]` is the ignored '[CLS]' token logit
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    print(f'Question: {question}')
    print(f'Answer: {answer}')
