# Inspird from Green Code
import requests
from bs4 import BeautifulSoup
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Constants
BASE_URL = "https://en.wikipedia.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Load model
modelPath = "./embedding_model"
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save(modelPath)
model = SentenceTransformer(modelPath)

# Initial and target pages
start_page = "/wiki/Potato"
target_page = "/wiki/Barack_Obama"

# Tracking variables
visited_pages = set()
start_time = time.time()

# Helper functions
def get_response(url):
    return requests.get(url, headers=HEADERS)

def parse_html(html):
    return BeautifulSoup(html, "html.parser")

def clean_title(wiki_path):
    return wiki_path.replace("/wiki/", "").replace("_", " ")

# Main logic
def find_first_valid_link(soup):
    texts = []
    hrefs = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)

        if (
            not href.startswith("/wiki/")
            or any(x in href for x in (":", "#", ".svg", "Main_Page"))
            or len(text) < 3
            or href in visited_pages
        ):
            continue

        texts.append(text)
        hrefs.append(href)

        if len(texts) >= 30:
            break

    if not texts:
        return None

    target_embedding = model.encode([clean_title(target_page)])
    embeddings = model.encode(texts)

    similarities = cosine_similarity(target_embedding, embeddings)[0]
    best_index = np.argmax(similarities)

    return hrefs[best_index]


current_page = start_page
cnt = 0
links = []

# Traversal loop
while current_page != target_page:
    print(f"\nVisiting: {BASE_URL + current_page}")
    visited_pages.add(current_page)

    try:
        res = get_response(BASE_URL + current_page)
        res.raise_for_status()
    except Exception as e:
        print(f"Request failed: {e}")
        break

    soup = parse_html(res.text)
    next_link = find_first_valid_link(soup)

    if not next_link:
        print("No valid link found.")
        break
    
    links.append(BASE_URL+next_link)
    # print(f"Next link: {next_link}")
    current_page = next_link

    cnt += 1

# Check pages
if current_page == target_page:
    print("\nTarget reached!")

else:
    print("\nTarget not reached.")

end_time = time.time()

# Summary
print("-------------------------------------------")
print(f"Overall Time: {end_time - start_time:.2f} seconds")
print(f"Total Links Visited: {cnt} links")
print("Average Time per Link: {:.2f} seconds".format((end_time - start_time) / cnt if cnt > 0 else 0))
print(f"Current Page: {BASE_URL + current_page}")
print(f"Links Traversed: ")
for i in links:
    print(i)
