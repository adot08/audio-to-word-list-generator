import json
import math
import nltk
import os
import re
import requests
import shutil

from nltk.stem import WordNetLemmatizer
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
from typing import Union, List

from config import config
nltk.data.path.append(config.nltk_data_path)
DATA_DIR = Path('./data')

def load_prompt(prompt_key, **params):
    # use cases
    # load_prompt('generate_response', query=query, context=context)
    # load_prompt('analyze_text', input_text=input_text)
    if prompt_key not in config.prompts:
        raise KeyError(f"Prompt '{prompt_key}' not found in config")

    prompt_path = os.path.join(os.path.dirname(__file__), config.prompts[prompt_key])
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, 'r') as file:
        item = file.read().strip()

    try:
        return item.format(**params)
    except KeyError as e:
        raise ValueError(f"Missing parameter for prompt '{prompt_key}': {str(e)}")

def process_dictionary(file_path):
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()

    entries = content.split('————————————')

    dictionary = {}
    for entry in entries:
        entry = entry.strip()
        if entry:
            lines = entry.split('\n')
            if lines:
                main_word_match = re.search(r'[★☆ ]+([\S]+)', lines[0].strip())
                if main_word_match:
                    main_word = main_word_match.group(1)
                    dictionary[main_word] = entry

                    # 处理派生词
                    for line in lines:
                        derived_word_match = re.search(r'^—(\S+)', line.strip())
                        if derived_word_match:
                            derived_word = derived_word_match.group(1)
                            # 移除词性后缀（如果存在）
                            derived_word = re.sub(r'[^\w\-]+.*$', '', derived_word)
                            dictionary[derived_word] = entry

    # 保存为JSON文件
    json_path = DATA_DIR / 'dictionary.json'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

    return dictionary

def read_hard_words(file_path):
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'\d+\.\s+(\S+)'
    hard_words = re.findall(pattern, content)

    return hard_words

def generate_hard_word_definitions(dictionary, hard_words):
    hard_word_definitions = []
    missed_words = []
    for word in hard_words:
        if word in dictionary:
            hard_word_definitions.append(f"{word}:\n{dictionary[word]}\n\n")
        else:
            missed_words.append(word)

    # definitions_path = DATA_DIR / 'hard_words_definitions.txt'
    # with definitions_path.open('w', encoding='utf-8') as f:
    #     f.writelines(hard_word_definitions)
    
    # missed_words_path = DATA_DIR / 'missed_words.txt'
    # with missed_words_path.open('w', encoding='utf-8') as f:
    #     f.writelines(word + '\n' for word in missed_words)
    return hard_word_definitions, missed_words


def load_wordlist(file_path):
    with file_path.open('r', encoding='utf-8') as f:
        return set(word.strip().lower() for word in f)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN  # 默认作为名词

def extract_difficult_words(text, wordlist):
    lemmatizer = WordNetLemmatizer()
    
    # 将文本转换为小写并分割成单词
    words = re.findall(r'\b\w+\b', text.lower())
    
    # 对单词进行词性标注
    pos_tags = nltk.pos_tag(words)
    
    # 找出不在词表中的单词（使用词形还原）
    difficult_words = set()
    for word, pos in pos_tags:
        lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos))
        if lemma not in wordlist and word not in wordlist:
            difficult_words.add(word)
    
    return difficult_words

def split_audio(file_path, chunk_size_mb=10, cache_dir=".cache"):
    """
    Split an audio file into chunks of specified size.
    
    :param file_path: Path to the audio file
    :param chunk_size_mb: Size of each chunk in MB
    :param cache_dir: Directory to store the chunks
    :return: List of paths to the chunk files
    """
    file_size = os.path.getsize(file_path)
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    num_chunks = math.ceil(file_size / chunk_size_bytes)
    
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    
    chunks = []
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    for i in tqdm(range(num_chunks)):
        start_byte = i * chunk_size_bytes
        end_byte = min((i + 1) * chunk_size_bytes, file_size)
        
        # Calculate time proportion
        start_ms = int((start_byte / file_size) * duration_ms)
        end_ms = int((end_byte / file_size) * duration_ms)
        
        chunk = audio[start_ms:end_ms]
        
        chunk_name = f"chunk_{i}.mp3"
        chunk_path = cache_path / chunk_name
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
    
    return chunks

def call_llm_api(messages: Union[str, List[str]], url: str, model_name: str, authorization: str, **kwargs):
    """
    Call the LLM API with the given parameters.

    :param messages: A string or a list of strings representing the conversation
    :param url: API endpoint URL
    :param model_name: Name of the model to use
    :param authorization: Authorization token
    :param kwargs: Additional parameters to customize the API call
    :return: API response as a Python object
    """
    # Format messages
    formatted_messages = format_messages(messages)

    # Default parameters
    default_params = {
        "stream": False,
        "max_tokens": 4000,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1
    }

    # Update default parameters with any provided kwargs
    default_params.update(kwargs)

    # Construct the payload
    payload = {
        "model": model_name,
        "messages": formatted_messages,
        **default_params
    }

    # Set up headers
    headers = {
        "Authorization": f"Bearer {authorization}",
        "Content-Type": "application/json"
    }

    # Make the API call
    response = requests.post(url, json=payload, headers=headers)

    # Check for successful response
    response.raise_for_status()

    # Parse and return the JSON response
    return response.json()

def format_messages(content: Union[str, List[str]]) -> List[dict]:
    """
    Format the input content into a list of message dictionaries.
    
    :param content: A string or a list of strings
    :return: A list of message dictionaries
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    
    if not isinstance(content, list):
        raise ValueError("Content must be either a string or a list of strings.")
    
    if len(content) % 2 == 0:
        raise ValueError("When providing a list, its length should be odd. The last message should be from the user.")
    
    messages = []
    for i, message in enumerate(content):
        if i % 2 == 0:
            role = "user"
        else:
            role = "assistant"
        messages.append({"role": role, "content": message})
    
    return messages

def transcribe_audio(file_path, authorization, url, model_name="FunAudioLLM/SenseVoiceSmall"):
    """
    Transcribe an audio file using the specified ASR API.
    
    :param file_path: Path to the audio file
    :param authorization: Authorization token for the API
    :param model_name: Name of the model to use
    :return: API response text
    """
    
    headers = {
        "Authorization": f"Bearer {authorization}",
    }
    
    files = {
        "file": open(file_path, "rb"),
    }
    
    data = {
        "model": model_name,
    }
    
    response = requests.post(url, headers=headers, files=files, data=data)
    return response.text

def transcribe_large_audio(file_path, authorization, url, model_name="FunAudioLLM/SenseVoiceSmall", chunk_size_mb=10):
    """
    Transcribe a large audio file by splitting it into chunks, transcribing each chunk,
    and then combining the results.
    
    :param file_path: Path to the audio file
    :param authorization: Authorization token for the API
    :param model_name: Name of the model to use
    :param chunk_size_mb: Size of each chunk in MB
    :return: Combined transcription result
    """
    chunks = split_audio(file_path, chunk_size_mb)
    # 虽然是io密集型的，但是和网络带宽也有关系，还是改成顺序执行吧
    transcriptions = []
    for chunk in tqdm(chunks):
        result = transcribe_audio(chunk, authorization, url, model_name)
        try:
            transcriptions.append(json.loads(result)['text'])
        except KeyError:
            print("warning! error occur in:", chunk)
    
    # Clean up cache
    shutil.rmtree(".cache")
    
    return " ".join(transcriptions)

def extract_words(text):
    pattern = r'\d+\.\s*(\w+)'
    matches = re.findall(pattern, text)
    return matches

def get_file_content(file_path):
    """
    读取文件内容，支持文本、音频和JSON文件。
    """
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension in ['txt', 'md', 'rtf']:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_extension in ['mp3', 'wav', 'ogg', 'flac']:
        return transcribe_large_audio(file_path, 
                                      authorization=config.api_key,
                                      url=config.asr_api_base_url,
                                      model_name=config.asr_model_name)
    elif file_extension == 'json':
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def get_hard_words(content, simple_word_list):
    """
    从内容中提取难词并使用 LLM 筛选。
    """
    hard_words = extract_difficult_words(content, simple_word_list)
    query = load_prompt("hard_words_selection", words_list=", ".join(hard_words))
    result = call_llm_api(query,
                          authorization=config.api_key,
                          url=config.llm_api_base_url,
                          model_name=config.llm_model_name)
    return extract_words(result["choices"][0]["message"]["content"])

def get_word_definitions(dictionary, words):
    """
    从字典中获取单词定义，并返回未找到定义的单词。
    """
    definitions = []
    missed_words = []
    for word in words:
        if word in dictionary:
            definitions.append(f"{word}\n{dictionary[word]}")
        else:
            missed_words.append(word)
    return definitions, missed_words

def get_missed_word_definitions(missed_words):
    """
    使用 LLM 获取未找到定义的单词的解释。
    """
    query = load_prompt("words_meaning_gen", words_list=", ".join(missed_words))
    result = call_llm_api(query,
                          authorization=config.api_key,
                          url=config.llm_api_base_url,
                          model_name=config.llm_model_name)
    return result["choices"][0]["message"]["content"]

def parse_definitions(definitions, is_ai_generated=False):
    """
    解析定义字符串，返回格式化的字典。
    """
    output = []
    for definition in definitions.split("\n\n"):
        splits = definition.split("\n", 1)
        word = splits[0].strip()
        content = splits[1].strip() if len(splits) > 1 else ""
        if " " in word:
            word_splits = word.split(" ", 1)
            word = word_splits[0]
            content = word_splits[1] + content
        key = "aiExplanation" if is_ai_generated else "definition"
        output.append({"word": word, key: content})
    return output

def file_to_desired_dict(file_path):
    """
    把一个文件转换成难词单词本
    """
    # 获取文件内容
    content = get_file_content(file_path)
    
    # 获取简单词列表
    simple_word_list = load_wordlist(DATA_DIR / config.simple_word_lists_name)
    
    # 获取难词列表
    true_hard_words = get_hard_words(content, simple_word_list)
    
    # 从字典中获取单词定义
    dictionary = process_dictionary(DATA_DIR / config.dict_file_name)
    hard_word_definitions, missed_words = get_word_definitions(dictionary, true_hard_words)
    
    # 获取未找到定义的单词的 AI 解释
    missed_words_definitions = get_missed_word_definitions(missed_words)
    
    # 解析并合并结果
    output = parse_definitions("\n\n".join(hard_word_definitions))
    output.extend(parse_definitions(missed_words_definitions, is_ai_generated=True))
    
    return output

def generate_example_sentence(word, selected_content):
    """
    为给定的单词和选定内容生成例句
    """
    query = load_prompt("example_generation", word=word, selected_content=selected_content)
    
    result = call_llm_api(query,
                          authorization=config.api_key,
                          url=config.llm_api_base_url,
                          model_name=config.llm_model_name)
    
    example = result["choices"][0]["message"]["content"]
    return example

if __name__ == "__main__":
    print(file_to_desired_dict("/Users/adot/Library/Group Containers/243LU875E5.groups.com.apple.podcasts/Library/Cache/980662BA-A3F7-46F6-977D-33F6CB349100.mp3"))