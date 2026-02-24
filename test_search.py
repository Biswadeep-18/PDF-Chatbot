from duckduckgo_search import DDGS

def test_search():
    query = "who is the current current cm of odisha"
    print(f"Searching for: {query}")
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                print(f"Title: {r['title']}")
                print(f"Snippet: {r['body']}")
                print("-" * 20)
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    test_search()
