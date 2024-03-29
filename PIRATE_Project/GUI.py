import tkinter as tk
from getDocument import *
from PdfUtils import *
from backend_GUI_functions import *
import threading
import math

def on_search_click():
    search_query = search_entry.get()
    retVal = update_displayed_text(search_query)

    if retVal is None:
        return
    # Create the new frame
    search_frame.pack_forget()
    results_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    print(f"Starting the PIRATE Score window with query: {search_query}\n Please wait while it loads")
    show_cited_papers_thread = threading.Thread(target=show_cited_papers_in_new_window, args=(search_query,))
    show_cited_papers_thread.start()

def on_text_click(event):
    index = text.index(f"@{event.x},{event.y}")
    tags = text.tag_names(index)
    if "section" in globals():
        global results_frame
        results_frame.destroy()
        results_frame = tk.Frame(root)
        results_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Do section comparsion
    print(f"Please wait while the section comparison completes, you clicked on section {math.floor(float(index) - 1)}")
    setResultFrame_thread = threading.Thread(target=setResultFrameWithSectionScores, args=(math.floor(float(index) - 1),))
    setResultFrame_thread.start()

def setResultFrameWithSectionScores(index):
    if("listOfCorpusComparison" not in globals()):
        print("Please wait until the cited paper's thread returns to click")
        return
    sectionSimilarityScores = fromSectionIndexGetTopRelevantSectionDetails(listOfCorpusComparison[:50], index)
    global section
    for i in sectionSimilarityScores:
        section = tk.Text(results_frame, wrap=tk.WORD, height=10)
        section.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5) # Adjust as needed
        section.insert(tk.END, f"Similarity: {i[0]} with paper title: {i[1][3]}\n")
        section.insert(tk.END, f"{i[1][2]}")

def update_displayed_text(search_query):
    text.delete(1.0, tk.END)

    returnValueFromSearch = fromQueryReturnText(search_query)

    if(returnValueFromSearch is None):
       text.insert(tk.END, "Search failed, please try again\n")
       return None

    for index, value in enumerate(returnValueFromSearch):
        text.insert(tk.END, value + "\n")
        text.tag_add("clickable", f"{index+1}.0", f"{index+1}.end")

    text.tag_config("clickable")
    text.bind("<Button-1>", on_text_click)

    return 1

def show_cited_papers_in_new_window(search_query):
    if(search_query is None):
        print(f"The cited paper search failed with query: {search_query}")
        return
    new_window = tk.Toplevel(root)
    new_window.title("Papers with their PIRATE Score")

    global textCitedWindow
    textCitedWindow = tk.Text(new_window, wrap=tk.WORD, yscrollcommand=scrollbar.set)
    textCitedWindow.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    textCitedWindow.tag_configure("red", foreground="red")
    textCitedWindow.tag_configure("green", foreground="green")

    global listOfCorpusComparison
    global depth
    listOfCorpusComparison, titles = fromQueryReturnCorpusComparison(search_query, depth)

    if(not listOfCorpusComparison):
        textCitedWindow.insert(tk.END, "Failed to get a corpus comparison, sorry")
        return
    
    for index, value in enumerate(listOfCorpusComparison):
        textCitedWindow.insert(tk.END, f"PIRATE Score: {value[0]:.3f} and it's {'not cited in ' if value[1] not in titles else 'cited in     '}the main paper with title: {value[1]:20s}\n")
        if value[1] in titles:
            textCitedWindow.tag_add("green", f"{1 + index}.28", f"{1 + index}.42")
        else:
            textCitedWindow.tag_add("red", f"{1 + index}.28", f"{1 + index}.42")

def splashInput():
    global depth
    depth = int(input_box.get())
    splash_screen.destroy()


if __name__ == '__main__':
    splash_screen = tk.Tk()
    splash_screen.title("Set depth!")

    splashText = tk.Text(splash_screen, wrap=tk.WORD)
    splashText.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    splashText.insert(tk.END, "Please insert an integer for depth, use 5 (or lower) for testing purposes and 100 to get all the papers. The value will dictate how many reference papers to use from the main paper to create the Reference Corpus. This is implemented only to give the user to understand the experince without the long loading times")
    
    input_box = tk.Entry(splash_screen, width=50)
    input_box.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=10)

    global depth
    submit_button = tk.Button(splash_screen, text="Submit", command=splashInput)
    submit_button.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=10, pady=10)

    splash_screen.mainloop()

    root = tk.Tk()
    root.title("Text Display with Interaction and Search")

    scrollbar = tk.Scrollbar(root)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text = tk.Text(root, wrap=tk.WORD, yscrollcommand=scrollbar.set)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar.config(command=text.yview)

    text.insert(tk.END, "Please search for a paper to begin, beware that your depth input determines loading times. A faster internet connection will be beneficial")

    # Create search entry and button
    search_frame = tk.Frame(root)
    search_frame.pack(side=tk.TOP, fill=tk.X)

    search_entry = tk.Entry(search_frame)
    search_entry.pack(side=tk.LEFT, fill=tk.X, expand=False)

    search_button = tk.Button(search_frame, text="Search", command=on_search_click)
    search_button.pack(side=tk.RIGHT)

    # Create the results frame which will display section comparisons
    global results_frame
    results_frame = tk.Frame(root)
    # scrollbarResults = tk.Scrollbar(results_frame)
    # scrollbarResults.pack(side=tk.RIGHT, fill=tk.Y)

    sections = []

    root.mainloop()