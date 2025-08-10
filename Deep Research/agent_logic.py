import json
from pathlib import Path
from copy import deepcopy
import tqdm.asyncio as ta
import asyncio

from llm_services import LLMService


async def summarize_articles(keyword: str, llm_service: LLMService, config: dict):
    """Generates and saves summaries for a list of full-text articles."""
    results_dir = Path(config["paths"]["results_dir"])
    full_articles_path = results_dir / f"full_articles_{keyword}.json"
    summaries_path = results_dir / f"summaries_{keyword}.json"

    if not full_articles_path.exists():
        print(f"No full articles file found for '{keyword}'. Skipping summarization.")
        return

    with open(full_articles_path, "r", encoding="utf-8") as f:
        full_article_list = json.load(f)

    tasks = []
    for article in full_article_list:
        content = article["Content"]
        # Truncate very long content to fit model context window
        if len(content) > 60000:
            content = content[:30000] + "\n\n... [Content truncated] ...\n\n" + content[-30000:]

        prompt = llm_service.get_prompt("article_summary", content=content)
        tasks.append(llm_service.call_llm(prompt))

    print(f"Summarizing {len(tasks)} articles for keyword '{keyword}'...")
    summaries = await ta.gather(*tasks)

    summary_list = []
    for article, summary in zip(full_article_list, summaries):
        item = deepcopy(article)
        item["Summary"] = summary
        summary_list.append(item)

    with open(summaries_path, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, ensure_ascii=False, indent=4)
    print(f"Summaries saved to {summaries_path}")


def _build_summary_context(summaries_path: Path) -> str:
    """Helper to build a concatenated string of summaries for context."""
    if not summaries_path.exists():
        return ""
    with open(summaries_path, "r", encoding="utf-8") as f:
        summary_list = json.load(f)

    context_blocks = []
    for i, record in enumerate(summary_list):
        doi_ref = f"[DOI: {record.get('DOI', 'N/A')}]"
        block = f"#{i + 1}: {record.get('Title', 'No Title')}\n{doi_ref}\nSummary: {record.get('Summary', 'No Summary')}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


async def generate_initial_outline(research_topic: str, keyword: str, llm_service: LLMService, config: dict) -> str:
    """Generates the first version of the research outline."""
    results_dir = Path(config["paths"]["results_dir"])
    summaries_path = results_dir / f"summaries_{keyword}.json"
    outline_path = results_dir / "initial_outline.md"

    summary_context = _build_summary_context(summaries_path)
    if not summary_context:
        print("No summary content available to generate outline. Aborting.")
        return ""

    prompt = llm_service.get_prompt(
        "outline_generation",
        research_topic=research_topic,
        summary_content=summary_context
    )

    print("Generating initial research outline...")
    initial_outline = await llm_service.call_llm(prompt)

    with open(outline_path, "w", encoding="utf-8") as f:
        f.write(initial_outline)
    print(f"Initial outline saved to {outline_path}")
    return initial_outline


async def reflect_and_identify_gaps(research_topic: str, current_outline: str, llm_service: LLMService,
                                    config: dict) -> str:
    """Uses the LLM to reflect on the outline and suggest a new keyword."""
    prompt = llm_service.get_prompt(
        "reflection_and_expansion",
        research_topic=research_topic,
        current_outline=current_outline
    )

    print("Reflecting on outline to find research gaps...")
    reflection_result = await llm_service.call_llm(prompt)

    # Clean up the keyword from the model's response
    keyword = reflection_result.strip().replace('"', '').replace('\n', ' ').strip()

    if not keyword:
        keyword = "TADF molecular design"  # Fallback keyword
        print(f"LLM did not suggest a keyword, using fallback: '{keyword}'")
    else:
        print(f"LLM suggested new research keyword: '{keyword}'")

    return keyword


async def expand_subtopic(research_topic: str, keyword: str, llm_service: LLMService, config: dict) -> str:
    """Generates a detailed article section for a subtopic."""
    results_dir = Path(config["paths"]["results_dir"])
    summaries_path = results_dir / f"summaries_{keyword}.json"

    summary_context = _build_summary_context(summaries_path)
    if not summary_context:
        print(f"No summaries for '{keyword}', cannot expand subtopic.")
        return ""

    prompt = llm_service.get_prompt(
        "expand_section",
        research_topic=research_topic,
        section_title=keyword,
        summary_content=summary_context
    )

    print(f"Expanding subtopic: '{keyword}'...")
    expanded_content = await llm_service.call_llm(prompt)

    # Save the expanded content for later assembly
    (results_dir / "expanded_sections").mkdir(exist_ok=True)
    section_path = results_dir / "expanded_sections" / f"{keyword.replace(' ', '_')}.md"
    with open(section_path, "w", encoding="utf-8") as f:
        f.write(expanded_content)

    print(f"Expanded content for '{keyword}' saved to {section_path}")
    return expanded_content


async def generate_final_report(research_topic: str, keywords: dict, contents: dict, config: dict):
    """Assembles the final research report from all generated parts."""
    results_dir = Path(config["paths"]["results_dir"])
    report_path = results_dir / "final_report_draft.md"

    # --- Assemble the report content ---
    full_report = f"# Research Report: {research_topic}\n\n"
    full_report += "## 1. Research Outline\n"
    full_report += contents['outline'] + "\n\n"

    full_report += f"## 2. Initial Research Overview: {keywords['initial']}\n"
    full_report += "(This section would contain the synthesized summary of the initial search)\n\n"

    full_report += f"## 3. Deep Dive: {keywords['reflection_1']}\n"
    full_report += contents['reflection_1'] + "\n\n"

    full_report += f"## 4. Deep Dive: {keywords['reflection_2']}\n"
    full_report += contents['reflection_2'] + "\n\n"

    # --- Assemble the references ---
    all_summaries = []
    for keyword in keywords.values():
        summary_file = results_dir / f"summaries_{keyword}.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                all_summaries.extend(json.load(f))

    seen_dois = set()
    unique_references = []
    for record in all_summaries:
        doi = record.get("DOI")
        if doi and doi not in seen_dois:
            seen_dois.add(doi)
            authors = ", ".join(record.get("Authors", []))
            ref = f"- **{record.get('Title', 'No Title')}** ({authors})\n  DOI: [{doi}](https://doi.org/{doi})"
            unique_references.append(ref)

    full_report += "## References\n" + "\n".join(unique_references)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"Draft of the final report saved to {report_path}")


async def polish_final_report(config: dict):
    """Sends the draft report to the LLM for final academic polishing."""
    results_dir = Path(config["paths"]["results_dir"])
    draft_path = results_dir / "final_report_draft.md"
    polished_path = results_dir / "final_report_polished.md"

    with open(draft_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    llm_service = LLMService(config)
    prompt = llm_service.get_prompt("report_polishing", report_content=report_content)

    print("Polishing the final report for academic quality...")
    polished_report = await llm_service.call_llm(prompt)

    with open(polished_path, "w", encoding="utf-8") as f:
        f.write(polished_report)
    print(f"Polished report saved to {polished_path}")