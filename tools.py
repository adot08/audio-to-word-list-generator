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
    """
    Load a prompt from a file and format it with the given parameters.

    Args:
        prompt_key (str): The key of the prompt in the config.
        **params: Additional parameters to format the prompt.

    Returns:
        str: The formatted prompt.

    Raises:
        KeyError: If the prompt key is not found in the config.
        FileNotFoundError: If the prompt file is not found.
        ValueError: If a required parameter is missing.

    Usage:
        load_prompt('generate_response', query=query, context=context)
        load_prompt('analyze_text', input_text=input_text)
    """
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
    """
    Process a dictionary file and create a JSON dictionary.

    Args:
        file_path (Path): The path to the dictionary file.

    Returns:
        dict: A dictionary containing processed entries.

    This function reads a dictionary file, processes its entries,
    and creates a JSON dictionary with main words and their derived forms.
    """
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

                    # Process derived words
                    for line in lines:
                        derived_word_match = re.search(r'^—(\S+)', line.strip())
                        if derived_word_match:
                            derived_word = derived_word_match.group(1)
                            # Remove part-of-speech suffix (if exists)
                            derived_word = re.sub(r'[^\w\-]+.*$', '', derived_word)
                            dictionary[derived_word] = entry

    # Save as JSON file
    json_path = DATA_DIR / 'dictionary.json'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

    return dictionary

def read_hard_words(file_path):
    """
    Read hard words from a file.

    Args:
        file_path (Path): The path to the file containing hard words.

    Returns:
        list: A list of hard words extracted from the file.
    """
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'\d+\.\s+(\S+)'
    hard_words = re.findall(pattern, content)

    return hard_words

def generate_hard_word_definitions(dictionary, hard_words):
    """
    Generate definitions for hard words from a dictionary.

    Args:
        dictionary (dict): A dictionary containing word definitions.
        hard_words (list): A list of hard words to look up.

    Returns:
        tuple: A tuple containing two lists:
               - List of string definitions for found words
               - List of words not found in the dictionary
    """
    hard_word_definitions = []
    missed_words = []
    for word in hard_words:
        if word in dictionary:
            hard_word_definitions.append(f"{word}:\n{dictionary[word]}\n\n")
        else:
            missed_words.append(word)

    return hard_word_definitions, missed_words

def load_wordlist(file_path):
    """
    Load a word list from a file.

    Args:
        file_path (Path): The path to the file containing the word list.

    Returns:
        set: A set of words from the file, converted to lowercase.
    """
    with file_path.open('r', encoding='utf-8') as f:
        return set(word.strip().lower() for word in f)

def get_wordnet_pos(treebank_tag):
    """
    Map the Penn Treebank POS tags to WordNet POS tags.

    Args:
        treebank_tag (str): A Penn Treebank POS tag.

    Returns:
        str: The corresponding WordNet POS tag.
    """
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN  # Default to noun

def extract_difficult_words(text, wordlist):
    """
    Extract words from the text that are not in the given wordlist.

    Args:
        text (str): The input text to process.
        wordlist (set): A set of known words.

    Returns:
        set: A set of words from the text that are not in the wordlist.
    """
    lemmatizer = WordNetLemmatizer()
    
    # Convert text to lowercase and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Perform POS tagging on the words
    pos_tags = nltk.pos_tag(words)
    
    # Find words not in the wordlist (using lemmatization)
    difficult_words = set()
    for word, pos in pos_tags:
        lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos))
        if lemma not in wordlist and word not in wordlist:
            difficult_words.add(word)
    
    return difficult_words

def split_audio(file_path, chunk_size_mb=10, cache_dir=".cache"):
    """
    Split an audio file into chunks of specified size.
    
    Args:
        file_path (str): Path to the audio file.
        chunk_size_mb (int): Size of each chunk in MB.
        cache_dir (str): Directory to store the chunks.

    Returns:
        list: List of paths to the chunk files.
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

    Args:
        messages (Union[str, List[str]]): A string or a list of strings representing the conversation.
        url (str): API endpoint URL.
        model_name (str): Name of the model to use.
        authorization (str): Authorization token.
        **kwargs: Additional parameters to customize the API call.

    Returns:
        dict: API response as a Python object.

    Raises:
        requests.exceptions.RequestException: If the API call fails.
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
    
    Args:
        content (Union[str, List[str]]): A string or a list of strings.

    Returns:
        List[dict]: A list of message dictionaries.

    Raises:
        ValueError: If the content is not a string or a list of strings, or if the list length is even.
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    
    if not isinstance(content, list):
        raise ValueError("Content must be either a string or a list of strings.")
    
    if len(content) % 2 == 0:
        raise ValueError("When providing a list, its length should be odd. The last message should be from the user.")
    
    messages = []
    for i, message in enumerate(content):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": message})
    
    return messages

def transcribe_audio(file_path, authorization, url, model_name="FunAudioLLM/SenseVoiceSmall"):
    """
    Transcribe an audio file using the specified ASR API.
    
    Args:
        file_path (str): Path to the audio file.
        authorization (str): Authorization token for the API.
        url (str): API endpoint URL.
        model_name (str): Name of the model to use.

    Returns:
        str: API response text.
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
    
    Args:
        file_path (str): Path to the audio file.
        authorization (str): Authorization token for the API.
        url (str): API endpoint URL.
        model_name (str): Name of the model to use.
        chunk_size_mb (int): Size of each chunk in MB.

    Returns:
        str: Combined transcription result.
    """
    chunks = split_audio(file_path, chunk_size_mb)
    transcriptions = []
    for chunk in tqdm(chunks):
        result = transcribe_audio(chunk, authorization, url, model_name)
        try:
            transcriptions.append(json.loads(result)['text'])
        except KeyError:
            print(f"Warning! Error occurred in: {chunk}")
    
    # Clean up cache
    shutil.rmtree(".cache")
    
    return " ".join(transcriptions)

def extract_words(text):
    """
    Extract words from numbered list in text.

    Args:
        text (str): Input text containing numbered list of words.

    Returns:
        list: List of extracted words.
    """
    pattern = r'\d+\.\s*(\w+)'
    matches = re.findall(pattern, text)
    return matches

def get_file_content(file_path):
    """
    Read file content, supporting text, audio, and JSON files.

    Args:
        file_path (str): Path to the file.

    Returns:
        Union[str, dict]: File content as string or dictionary for JSON files.

    Raises:
        ValueError: If the file type is unsupported.
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
    Extract difficult words from content and filter them using LLM.

    Args:
        content (str): Input content to extract words from.
        simple_word_list (set): Set of simple words to compare against.

    Returns:
        list: List of hard words selected by LLM.
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
    Retrieve word definitions from a dictionary and identify missing words.

    Args:
        dictionary (dict): Dictionary containing word definitions.
        words (list): List of words to look up.

    Returns:
        tuple: A tuple containing:
            - list: Definitions found in the dictionary.
            - list: Words not found in the dictionary.
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
    Use LLM to generate definitions for words not found in the dictionary.

    Args:
        missed_words (list): List of words without definitions.

    Returns:
        str: LLM-generated definitions for the missed words.
    """
    query = load_prompt("words_meaning_gen", words_list=", ".join(missed_words))
    result = call_llm_api(query,
                          authorization=config.api_key,
                          url=config.llm_api_base_url,
                          model_name=config.llm_model_name)
    return result["choices"][0]["message"]["content"]

def parse_definitions(definitions, is_ai_generated=False):
    """
    Parse definition strings and return formatted dictionaries.

    Args:
        definitions (str): String containing word definitions.
        is_ai_generated (bool): Flag indicating if definitions are AI-generated.

    Returns:
        list: List of dictionaries containing parsed word definitions.
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
    Convert a file to a dictionary of difficult words with their definitions.

    Args:
        file_path (str): Path to the input file.

    Returns:
        list: List of dictionaries containing words and their definitions/explanations.
    """
    # Get file content
    content = get_file_content(file_path)
    
    # Get simple word list
    simple_word_list = load_wordlist(DATA_DIR / config.simple_word_lists_name)
    
    # Get list of difficult words
    true_hard_words = get_hard_words(content, simple_word_list)
    
    # Get word definitions from dictionary
    dictionary = process_dictionary(DATA_DIR / config.dict_file_name)
    hard_word_definitions, missed_words = get_word_definitions(dictionary, true_hard_words)
    
    # Get AI-generated explanations for words not found in the dictionary
    missed_words_definitions = get_missed_word_definitions(missed_words)
    
    # Parse and merge results
    output = parse_definitions("\n\n".join(hard_word_definitions))
    output.extend(parse_definitions(missed_words_definitions, is_ai_generated=True))
    
    return output

def generate_example_sentence(word, selected_content):
    """
    Generate an example sentence for a given word based on selected content.

    Args:
        word (str): The word to generate an example for.
        selected_content (str): Content to base the example on.

    Returns:
        str: Generated example sentence.
    """
    query = load_prompt("example_generation", word=word, selected_content=selected_content)
    
    result = call_llm_api(query,
                          authorization=config.api_key,
                          url=config.llm_api_base_url,
                          model_name=config.llm_model_name)
    
    example = result["choices"][0]["message"]["content"]
    return example