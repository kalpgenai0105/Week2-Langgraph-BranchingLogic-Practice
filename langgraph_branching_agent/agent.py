"""LangGraph agent with branching logic.

            +------------+
            | classifier |
            +------------+
              /        \
     technical          business
        |                  |
  tech_report          biz_report
        |                  |
       END                END
"""
import argparse
import os
import re
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME", "gemini-3.8-flash"), temperature=0)


class State(TypedDict):
    topic: str
    category: Literal["technical", "business"]
    report: str


def classifier(state: State) -> dict:
    """Decide whether the topic is technical or business."""
    prompt = (
        "Classify the following topic as either 'technical' or 'business'. "
        "Reply with exactly one word.\n\nTopic: " + state["topic"]
    )
    answer = llm.invoke(prompt).text.strip().lower()
    return {"category": "technical" if "technical" in answer else "business"}


def route(state: State) -> str:
    """Conditional edge: pick the next node from the classification."""
    return state["category"]


def tech_report(state: State) -> dict:
    prompt = (
        "Write a concise technical report on the topic below. Include: Overview, "
        "Architecture / How it works, Key technologies, Challenges, Recommendations.\n\n"
        f"Topic: {state['topic']}"
    )
    return {"report": llm.invoke(prompt).text}


def biz_report(state: State) -> dict:
    prompt = (
        "Write a concise business report on the topic below. Include: Executive summary, "
        "Market opportunity, Costs and ROI, Risks, Strategic recommendations.\n\n"
        f"Topic: {state['topic']}"
    )
    return {"report": llm.invoke(prompt).text}


def build_graph():
    g = StateGraph(State)
    g.add_node("classifier", classifier)
    g.add_node("tech_report", tech_report)
    g.add_node("biz_report", biz_report)

    g.add_edge(START, "classifier")
    g.add_conditional_edges(
        "classifier", route, {"technical": "tech_report", "business": "biz_report"}
    )
    g.add_edge("tech_report", END)
    g.add_edge("biz_report", END)
    return g.compile()


graph = build_graph()


def save_report(topic: str, category: str, report: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60] or "report"
    path = Path(__file__).parent / "reports" / f"{category}-{slug}.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"# {topic}\n\n_Category: {category}_\n\n{report}\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a technical or business report.")
    parser.add_argument("topic", nargs="*", help="Report topic (prompted if omitted)")
    parser.add_argument("-s", "--save", action="store_true", help="Save the report to reports/*.md")
    args = parser.parse_args()

    topic = " ".join(args.topic) or input("Enter a topic: ")
    result = graph.invoke({"topic": topic})
    print(f"\n[Category: {result['category']}]\n")
    print(result["report"])
    if args.save:
        print(f"\nSaved to {save_report(topic, result['category'], result['report'])}")
