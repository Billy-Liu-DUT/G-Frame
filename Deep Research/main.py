import asyncio
import json
import yaml
from pathlib import Path

from agent_logic import (
    summarize_articles,
    generate_initial_outline,
    reflect_and_identify_gaps,
    expand_subtopic,
    generate_final_report,
    polish_final_report,
)
from data_retrieval import retrieve_and_process_articles
from llm_services import LLMService


async def main_workflow():
    """Orchestrates the main workflow for the research agent."""
    # --- 1. Load Configuration ---
    print("Loading configuration from config.yaml...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Create results directory if it doesn't exist
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(exist_ok=True, parents=True)

    # --- 2. Initialize Services ---
    llm_service = LLMService(config)

    # --- 3. User Input ---
    research_topic = input("Please enter your research topic (e.g., 'TADF Material Design'): ")
    initial_keyword = input("Please enter an initial keyword for Web of Science (e.g., 'TADF'): ")

    # --- 4. Initial Research Round ---
    print(f"\n--- Starting Initial Research Round for keyword: '{initial_keyword}' ---")
    await retrieve_and_process_articles(initial_keyword, config)
    await summarize_articles(initial_keyword, llm_service, config)
    initial_outline = await generate_initial_outline(research_topic, initial_keyword, llm_service, config)

    # --- 5. First Reflection and Deep Dive ---
    print("\n--- Reflecting on initial findings to identify research gaps... ---")
    reflection_keyword_1 = await reflect_and_identify_gaps(research_topic, initial_outline, llm_service, config)

    print(f"\n--- Starting 1st Deep Dive for keyword: '{reflection_keyword_1}' ---")
    await retrieve_and_process_articles(reflection_keyword_1, config)
    await summarize_articles(reflection_keyword_1, llm_service, config)
    reflection_content_1 = await expand_subtopic(research_topic, reflection_keyword_1, llm_service, config)

    # --- 6. Second Reflection and Deep Dive ---
    print("\n--- Reflecting again to identify further research gaps... ---")
    reflection_keyword_2 = await reflect_and_identify_gaps(research_topic, initial_outline, llm_service, config)

    print(f"\n--- Starting 2nd Deep Dive for keyword: '{reflection_keyword_2}' ---")
    await retrieve_and_process_articles(reflection_keyword_2, config)
    await summarize_articles(reflection_keyword_2, llm_service, config)
    reflection_content_2 = await expand_subtopic(research_topic, reflection_keyword_2, llm_service, config)

    # --- 7. Final Report Generation ---
    print("\n--- Generating and Polishing Final Research Report ---")
    keywords = {
        "initial": initial_keyword,
        "reflection_1": reflection_keyword_1,
        "reflection_2": reflection_keyword_2,
    }
    contents = {
        "outline": initial_outline,
        "reflection_1": reflection_content_1,
        "reflection_2": reflection_content_2,
    }
    await generate_final_report(research_topic, keywords, contents, config)
    await polish_final_report(config)

    print("\n✅ Workflow complete! Check the 'results' directory for the final report.")


if __name__ == "__main__":
    try:
        asyncio.run(main_workflow())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")