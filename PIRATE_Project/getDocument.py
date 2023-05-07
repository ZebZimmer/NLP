# https://www.semanticscholar.org/product/api
# Above is the complete set of API calls
# chatGPT was used as a tool to help do this. I forgot many of the things I learned in CSCI 4131

import requests
import json
import torch
import concurrent.futures

Paper_to_search_for = "BERT Rediscovers the Classical NLP Pipeline"

params = {
    "api_key": "Wz3YEvwO1c5VRXaETxxmK6bWSxmWLTzZ6GrTvD5e"
}

# Input the title (as exact as possible) of the paper 
# returns the paper's ID in the S2ORC database
def get_paperId_from_query(paper_to_search_for: str) -> str:
    if paper_to_search_for is None:
        return None
    url = 'https://api.semanticscholar.org/graph/v1/paper/search?query=' + paper_to_search_for

    response = requests.get(url, params=params)

    if response.status_code == 200:
        try:
            content = json.loads(response.content.decode("utf-8"))["data"]
        except Exception as e:
            return None
    else:
        return None

    return content[0]["paperId"]

# Input the paperId in str form and get the embedding and title of the paper
# Specifically the SPECTER embedding (https://arxiv.org/abs/2004.07180)
def get_paper_embedding_and_title_from_paperId(paperId_to_search_with: str) -> tuple[list, list]:
    url = 'https://api.semanticscholar.org/graph/v1/paper/' + paperId_to_search_with + '?fields=embedding,title'

    embedding = ""
    title = ""

    response = requests.get(url, params=params)

    if response.status_code == 200:
        # print(response.content)
        embedding = json.loads(response.content.decode("utf-8"))["embedding"]["vector"]
        title = json.loads(response.content.decode("utf-8"))["title"]

    else:
        print('Request failed.')
        print(response.content)
    
    return (embedding, title)

# Input the paperId in str form and get it's embedding as well as the embeddings of all referenced papers
def get_paper_and_references_embedding_and_titles_from_paperId(paperId_to_search_with: str) -> tuple[list, list]:
    if(paperId_to_search_with is None):
        return None
    url = 'https://api.semanticscholar.org/graph/v1/paper/' + paperId_to_search_with + '?fields=references,embedding,title'

    embeddings = []
    references = ""
    titles = []

    response = requests.get(url, params=params)
    if response.status_code == 200:
        # print(response.content)
        embeddings.append(json.loads(response.content.decode("utf-8"))["embedding"]["vector"])
        references = json.loads(response.content.decode("utf-8"))["references"]
        titles.append(json.loads(response.content.decode("utf-8"))["title"])
    else:
        print('Request failed.')
        print(response.content)

    # for i in range(len(references) - 1):
    #     if references[i]["paperId"]:
    #         temp = get_paper_embedding_and_title_from_paperId(references[i]["paperId"])
    #         embeddings.append(temp[0])
    #         titles.append(temp[1])

    return get_embeddings_and_titles(references)

# GPT4 helped me create this function that may speed up the searches
def get_embeddings_and_titles(references):
    def fetch_data(index, paper_id, embeddings, titles):
        temp = get_paper_embedding_and_title_from_paperId(paper_id)
        embeddings[index] = temp[0]
        titles[index] = temp[1]

    embeddings = [None] * (len(references) - 1)
    titles = [None] * (len(references) - 1)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fetch_data, i, references[i]["paperId"], embeddings, titles)
            for i in range(len(references) - 1)
            if references[i]["paperId"]
        ]
        concurrent.futures.wait(futures)

    return (embeddings, titles)

# Main call that returns a list of the embeddings and titles from a single query
# The first value in each list is of the main paper
def get_paper_and_references_embedding_and_titles_from_query(paper_to_search_for: str) -> tuple[list, list]:
    temp1 = get_paperId_from_query(paper_to_search_for)
    return get_paper_and_references_embedding_and_titles_from_paperId(temp1)