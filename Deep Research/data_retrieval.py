import asyncio
import json
from pathlib import Path
from typing import List, Dict
import tqdm
import clarivate.wos_starter.client
from clarivate.wos_starter.client.rest import ApiException
from playwright.async_api import async_playwright


def _get_wos_articles(query: str, config: dict) -> List[Dict]:
    """Helper function to retrieve articles from the Web of Science API."""
    wos_config = config['wos_search_params']
    api_key = config['api_keys']['clarivate_wos']

    api_config = clarivate.wos_starter.client.Configuration(
        host="https://api.clarivate.com/apis/wos-starter/v1"
    )
    api_config.api_key['ClarivateApiKeyAuth'] = api_key

    query = query.replace('\n', ' ').replace('"', ' ').strip()

    try:
        with clarivate.wos_starter.client.ApiClient(api_config) as api_client:
            api_instance = clarivate.wos_starter.client.DocumentsApi(api_client)
            response = api_instance.documents_get(
                q=query,
                db=wos_config['database'],
                limit=wos_config['limit_per_query'],
                page=1,
                sort_field=wos_config['sort_field'],
                detail='full'
            )

            def extract_info(doc):
                return {
                    "Title": doc.title,
                    "DOI": doc.identifiers.doi if doc.identifiers else None,
                    "Authors": [a.display_name for a in doc.names.authors] if doc.names else [],
                }

            return [extract_info(doc) for doc in response.hits]
    except ApiException as e:
        print(f"Web of Science API Error: {e.status} - {e.reason}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during WOS search: {e}")
        return []


async def _scrape_full_text(doi: str) -> str:
    """Scrapes the full text of an article given its DOI using Playwright."""
    if not doi:
        return ""
    url = f"https://doi.org/{doi}"
    try:
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=60000)

            # A simple heuristic to get the main content text
            main_content = await page.evaluate('''() => {
                const main = document.querySelector('article') || document.querySelector('div[role="main"]') || document.body;
                return main.innerText;
            }''')

            await browser.close()
            return main_content
    except Exception as e:
        print(f"Failed to scrape {url}. Error: {e}")
        return ""


async def retrieve_and_process_articles(keyword: str, config: dict):
    """
    Main data retrieval function.
    1. Gets article metadata from WOS.
    2. Scrapes full text for each article.
    """
    results_dir = Path(config["paths"]["results_dir"])

    # Step 1: Get article list from Web of Science
    print(f"Retrieving article metadata for keyword: '{keyword}'...")
    # This query structure can be expanded in the config
    query_expression = f'PY=2024 AND TS="{keyword}"'
    articles = _get_wos_articles(query_expression, config)

    if not articles:
        print("No articles found for this keyword.")
        return

    print(f"Found {len(articles)} articles. Now retrieving full text...")

    # Step 2: Scrape full text for each article with a valid DOI
    tasks = [_scrape_full_text(article["DOI"]) for article in articles]

    full_texts = []
    for f in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scraping Full Texts"):
        full_texts.append(await f)

    # Step 3: Combine metadata with full text and save
    full_article_list = []
    for article, text in zip(articles, full_texts):
        if text:  # Only include articles where full text was successfully retrieved
            article["Content"] = text
            full_article_list.append(article)

    output_path = results_dir / f"full_articles_{keyword}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_article_list, f, ensure_ascii=False, indent=4)

    print(f"Successfully retrieved and saved {len(full_article_list)} full-text articles to {output_path}")