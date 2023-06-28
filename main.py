from fastapi import FastAPI, Request
from os import getenv
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()


@app.post("/")
async def model(request: Request):
    json_body = await request.json()
    comment = json_body.get("comment")  # get value of "comment" key

    main_main = pd.read_csv('moaz_texts_dataset_20000.csv')
    text = 'this corse is level 3 informatics'
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)

    output = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=model.config.eos_token_id,
                            max_length=600, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(text)
    # print(generated_text)
    input_text = generated_text
    input_text = input_text.lower()
    tokens = word_tokenize(input_text)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    preprocessed_text = ' '.join(stemmed_tokens)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(main_main['full_text'])
    input_vector = tfidf_vectorizer.transform([preprocessed_text])

    cosine_similarities = cosine_similarity(input_vector, tfidf_vectorizer.transform(main_main['full_text']))
    most_similar_index = cosine_similarities.argmax()
    most_similar_text = main_main['full_text'][most_similar_index]

    # print("Most similar text:", most_similar_text)
    # print("Most similar text:", most_similar_index)
    # print(main_main['title_y'][most_similar_index])

    main_main_replay = pd.read_csv('moaz_full_eng_dataset.csv')

    title_x = main_main_replay['title_x'][most_similar_index]
    title_y = main_main_replay['title_y'][most_similar_index]
    kind = main_main_replay['kind'][most_similar_index]
    description_x = main_main_replay['description_x'][most_similar_index]
    channel = main_main_replay['channel'][most_similar_index]
    category = main_main_replay['category'][most_similar_index]
    language_x = main_main_replay['language_x'][most_similar_index]
    description_y = main_main_replay['description_y'][most_similar_index]
    text = main_main_replay['text'][most_similar_index]
    language_y = main_main_replay['language_y'][most_similar_index]
    copyright_holder = main_main_replay['copyright_holder'][most_similar_index]
    license = main_main_replay['license'][most_similar_index]
    level = main_main_replay['level'][most_similar_index]

    # json response
    json_response = {
        "result": "ok",
        "title_x": title_x,
        "title_y": title_y,
        "kind": kind,
        "description_x": description_x,
        "channel": channel,
        "category": category,
        "language_x": language_x,
        "description_y": description_y,
        "text": text,
        "language_y": language_y,
        "copyright_holder": copyright_holder,
        "license": license,
        "level": level,
    }
    return json_response


if __name__ == "__main__":
    port = int(getenv("PORT", 8000))
    uvicorn.run("main:app",
                # host="0.0.0.0",
                port=port, reload=True)