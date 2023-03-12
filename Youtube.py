from numpy.linalg import norm
from time import time, sleep
from pytube import YouTube
import numpy as np
import textwrap
import dotenv
import openai
import json
import os
import re

dotenv.load_dotenv('.env')
openai.api_key = os.getenv('API_KEY')


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def save_file(file_path, input):
    with open(file_path, 'w') as file:
        file.write(input)


def open_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def append_file(file_path, input):
    with open(file_path, 'a') as file:
        file.write(input)


def similarity(vector1, vector2):
    return np.dot(vector1,vector2)/(norm(vector1)*norm(vector2))


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

def gpt3_completion(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    reply = response["choices"][0]["text"].strip()
    return reply

def gpt35_completion(messages, model='gpt-3.5-turbo'):
    # token limitation of turbo is 4096
    response = openai.ChatCompletion.create(model=model, messages=messages)
    text = response['choices'][0]['message']['content']
    file_name = str(time()) + '_chat.txt'
    save_file(f'./chat_log/{file_name}', str(messages) + '\n\n==========\n\n' + text)
    return text


def save_embedding(text_path, index_path):
    chunks = ''
    all_text = open_file(text_path)
    chunks = textwrap.wrap(all_text, 500) # set video info chunks to 500 words
    result = []
    for chunk in chunks:
        embedding = gpt3_embedding(chunk.encode(encoding='ASCII',errors='ignore').decode())
        info = {'content': chunk, 'vector': embedding}
        result.append(info)
        sleep(2) # OpenAI ratelimit is 1 request per second
        print('creating embedding...')
    print('embedding finished')
    save_json(index_path, result)


def whisper(file_path, file_name):
    start_time = time()
    audio_file= open(file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    text_path = './text_files/' + file_name[:-4] + '.txt'
    save_file(text_path, str(transcript))
    print(f'Convertion finished in: {time() - start_time} seconds')
    return text_path


def fetch_audio_from_youtube(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    destination = './audio_files'
    out_file = video.download(output_path=destination)
    base, ext = os.path.splitext(out_file)
    # pattern = re.compile('[^a-zA-Z0-9]')
    # base = pattern.sub('', base)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    file_name = os.path.basename(new_file)
    print(yt.title + "\nhas been successfully downloaded.")
    return file_name


def flatten_convo(conversation):
    convo = ''
    for i in conversation:
        convo += '%s: %s\n' % (i['role'].upper(), i['content'])
    return convo.strip()


def fetch_relevant_info(text, embedding_path, count):
    scores = []
    vector = gpt3_embedding(text)
    embedding = load_json(embedding_path)
    for i in embedding:
        score = similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    rank = sorted(scores, key=lambda d: d['score'], reverse=True)
    top_similarity = rank[:count]
    return top_similarity


def generate_summary(embedding_path):
    embedding_list = load_json(embedding_path)
    summary = ''
    pre_prompt_1 = '\nPls give me a summary of info above.'
    pre_prompt_2 = '\nPls give me a brief summary of these paragraphs above. They are some decriptions of a youtube video, so pls start with phrases similar to: \'this video mainly about, but don\'t be exactly the same.'
    for i in embedding_list:
        content = i['content']
        prompt = content + pre_prompt_1
        completion = gpt3_completion(prompt)
        summary += completion
        print('generating summary...')
        sleep(2)
    final_prompt = summary + pre_prompt_2
    final_summary = gpt3_completion(final_prompt)
    return final_summary

def main(url):
    # it's file_name of mp3: "xxxxx.mp3"
    file_name = fetch_audio_from_youtube(url)
    mp3_path = f'audio_files/{file_name}'
    print('Whisper start to converting audio to text...')
    text_path = whisper(mp3_path, file_name)
    json_path = './embedding_files/' + file_name[:-4] + '.json'
    save_embedding(text_path, json_path)
    
    summary = generate_summary(json_path)
    print(f'\n\nSummary of {file_name} is:\n{summary}')
    conversation = []
    conversation.append({'role': 'system', 'content': 'I am an AI assistant named Whisper. My goal is answer user\'s questions about relevant infomation of some youtube videos contents.'})
    pre_prompt = 'This is relevant info from the video: '
    while True:
        question = input('\nUSER: ')
        top_similarity_embedding_list = fetch_relevant_info(question, json_path, 3)
        relevant_content = ''
        for embedding in top_similarity_embedding_list:
            # starts from 13th character to ignore "{   "text":"
            relevant_content += embedding['content'][13:]
        conversation.append({'role': 'user', 'content':pre_prompt + relevant_content + question})
        try:
            completion = gpt35_completion(conversation)
        except (openai.error.RateLimitError, openai.error.RateLimitError) as e:
            print('Sorry, openai is busy', e)
            break
        except openai.error.InvalidRequestError as e:
            print('Token limitation reached', e)
            break
        print('\nWhisper: ' + completion)
        conversation.append({'role': 'assistant', 'content': completion})


"""
[
{'role': 'system', 'content': 'hi i am chatgpt'}, 
{'role': 'user', 'content': preprompt + 'can you help me answer some questions'}, 
{'role': 'assistant', 'content': 'of course,  happy to help'}
]



[{'role': 'system', 'content': 'hi i am chatgpt'}, {'role': 'user', 'content': 'can you help me answer some questions'}, {'role': 'system', 'content': 'of course,  happy to help'}]

>>> flatten_convo(conversation)
'SYSTEM: hi i am chatgpt\nUSER: can you help me answer some questions\nSYSTEM: of course,  happy to help'
"""