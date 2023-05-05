import requests
import fitz

#get the url to the pdf
#will return None if pdf can't be obtained
def get_pdf_url(paper_id) -> str:
    url = 'https://api.semanticscholar.org/graph/v1/paper/' + paper_id + '?fields=openAccessPdf,title'
    response = requests.get(url)

    if response.status_code == 200:
        # content = json.loads(response.content.decode("utf-8"))["data"]
        response_json = response.json()
        # print(response_json)
        if "openAccessPdf" in response_json:
            if response_json["openAccessPdf"] != None:
                raw_text_url = response_json["openAccessPdf"]["url"]
                # print(raw_text_url)
                return raw_text_url
            else:
                return None
        else:
            print('no open access')
            return None 
    else:
        return None

def get_pdf_text_sections(link) -> list:
    retVal = []
    try:
        pdf_response = requests.get(link, timeout=8)#this may take awhile if bad link
    except Exception:
        print("Timeout")
        return None
    
    if pdf_response.status_code == 200:
        # print('ok link')
        # Save the content of the response to a file
        with open("sample.pdf", "wb") as f:
            f.write(pdf_response.content)
    else:
        return None
    
    doc = fitz.open('sample.pdf')
    for page in doc:
        text = page.get_text("blocks")  # get plain text
        # print(text[0])
        for elem in text:
            to_add = elem[4]
            to_add = to_add.replace('-\n', '')
            to_add = to_add.replace('\n', ' ')
            retVal.append(to_add)
    return retVal

def get_text(paper_id) -> list:
        '''FOR USE
        Returns a list of the sections in the pdf document
        Requires the paper ID string'''
        if(paper_id):
            link = get_pdf_url(paper_id)
            if(link == None):
                return None
            retVal = get_pdf_text_sections(link)
            return retVal
        else:
            return None