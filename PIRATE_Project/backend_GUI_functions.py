from getDocument import *
from PdfUtils import *
from sentence_transformers import SentenceTransformer

# A function to return just the first value of the list to be used in list.sort()
def getKey(listOfCosAndTitle):
    return listOfCosAndTitle[0]

# A function to return just the first value of the list to be used in list.sort()
def getKeySecondSort(listOfCosAndValue):
    return listOfCosAndValue[0]

# From a query string return the text of the paper as a list where each value is a section of the paper
def fromQueryReturnText(searchQuery: str) -> list:
    # mainPaper = get_paper_embedding_and_title_from_paperId(get_paperId_from_query(searchQuery))
    mainPaperSections = get_text(get_paperId_from_query(searchQuery))
    return mainPaperSections

# From a query string return a tuple
# tuple[0] is the sorted list of reference corpus with tuple[0][0] as the cosine similarity to main paper and tuple[0][1] is the title of the compared paper
# tuple[1] is a list of the titles from the main paper and its references 
def fromQueryReturnCorpusComparison(searchQuery: str) -> tuple((list[list], list)):
    defaultPaper = get_paper_and_references_embedding_and_titles_from_query(searchQuery)

    newCosineSimilarity = torch.nn.CosineSimilarity(dim=0)

    # Create a dictionary for the purpose of not repreating titles
    title_embedding_dict = {}

    listOfSearchedPapers = [] # For speed purposes, took 4min 20sec before

    for embedding, title in zip(defaultPaper[0], defaultPaper[1][:5]): #TODO remove this cap
        if title not in listOfSearchedPapers:
            listOfSearchedPapers.append(title)
            subPaper = get_paper_and_references_embedding_and_titles_from_query(title)
            # First depth, all of the main paper's references
            if title not in title_embedding_dict:
                title_embedding_dict[title] = embedding

            # Second depth, all of the referenced paper's references
            for sub_embedding, sub_title in zip(subPaper[0], subPaper[1]):
                if sub_title not in title_embedding_dict:
                    title_embedding_dict[sub_title] = sub_embedding

    mainPaperEmbedding = torch.tensor(defaultPaper[0][0])

    similarityScoresWithTitle = []
    # Get the cos similarity between the main paper and all of the subpapers
    for key in title_embedding_dict.keys():
        compared_paper = torch.tensor(title_embedding_dict[key])
        similarityScoresWithTitle.append([newCosineSimilarity(mainPaperEmbedding, compared_paper), key])

    similarityScoresWithTitle.sort(reverse=True, key=getKey)

    return (similarityScoresWithTitle, defaultPaper[1])

# This is a work in progress, find out why the inputted index is throwing errors for being out of bounds
# likely it's that the actual search into the object with indexOfSection is done incorrectly
def fromSectionIndexGetTopRelevantSectionDetails(similarityScoresWithTitle: list[list[tuple]], indexOfSection: int) -> list[list[int, int, list[list, int, str, str]]]:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = {}
    #Try parsing the top 10 things (that will include the main paper) and then see how long it takes to get and parse the pdfs
    totalCount = 0
    goodPDFCount = 0
    for i in similarityScoresWithTitle:
        listOfSections = get_text(get_paperId_from_query(i[1]))
        if(listOfSections):
            goodPDFCount += 1
            embeddings[i[1]] = []
            for index, value in enumerate(listOfSections):
                # print(f"{type(i[1])} and {type(index)} and {type(listOfSections)} and {type(value)}")
                embeddings[i[1]].append((model.encode(value), index, value, i[1]))
        totalCount += 1

    print(f"The number of good PDF downloads was {goodPDFCount} out of the {totalCount} tries so {goodPDFCount/totalCount}%")

    newCosineSimilarity = torch.nn.CosineSimilarity(dim=0)
    sectionSimilarityScores = []

    for key, value in embeddings.items():
        if(key == similarityScoresWithTitle[0][1]): # This is the title of the main paper
            continue
        for i in range(len(value)):
            # Appends the comparison of main paper's section and the comparing section, then all of the metadata of that comparing section for later refernce if necessary
            sectionSimilarityScores.append([newCosineSimilarity(torch.tensor(embeddings[similarityScoresWithTitle[0][1]][int(indexOfSection)][0]),  torch.tensor(value[i][0])), value[i]])

    # Sort the sections 
    sectionSimilarityScores.sort(reverse=True, key=getKeySecondSort)

    return sectionSimilarityScores

