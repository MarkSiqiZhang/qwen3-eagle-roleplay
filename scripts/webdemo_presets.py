"""
Canned prompts for the side-by-side web demo.

Two preset domains are surfaced:
  * In-domain (Jack Sparrow) — pulled from data/roleplay_data/jack_sparrow_test.jsonl.
  * Out-of-domain (UltraChat general chat) — pulled from
    data/roleplay_data/ultrachat_test.jsonl.

System prompts match the exact strings used by the benchmark harness so the
on-screen TPS numbers line up with results/RESULTS.md.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "roleplay_data"

JACK_SPARROW_SYSTEM = (
    "You are Jack Sparrow. Stay in character at all times and respond as Jack "
    "Sparrow would.\n\n"
    "Character description: A charming and eccentric pirate with a love for rum "
    "and adventure, you are known for your witty remarks, flamboyant style, and "
    "unpredictable behavior. Having sailed the seas for years, you have "
    "encountered numerous enemies and allies, constantly navigating treacherous "
    "waters in search of treasure. Despite your reputation as a trickster, you "
    "often find yourself entangled in dangerous situations that test your "
    "cunning and resourcefulness. From your iconic swagger to your clever "
    "schemes, your journey is filled with thrilling escapades and unexpected "
    "twists as you strive to outwit your adversaries and reclaim your lost "
    "glory. Your catchphrase is: \"Why is the rum always gone?\""
)

ULTRACHAT_SYSTEM = "You are a helpful assistant."

CURATED_JACK_SPARROW_PROMPTS = [
    "Captain, the crew is getting restless about the rum rations. What do you say to them?",
    "A Royal Navy frigate has us cornered off the Tortuga coast. What's the plan?",
    "Tell me the story of how you first became captain of the Black Pearl.",
    "Someone just stole your compass. How do you get it back?",
    "A mysterious stranger offers you a map to Davy Jones' locker. Do you take it?",
    "What advice would you give to a young sailor who wants to become a pirate?",
]

CURATED_ULTRACHAT_PROMPTS = [
    "Explain the difference between supervised and unsupervised machine learning, with an example of each.",
    "Write a short Python function that checks whether a string is a palindrome.",
    "What are the fundamental principles of Taoism, and how have they influenced Eastern philosophy?",
    "I'm planning a trip to Tokyo in April. What are three neighborhoods I shouldn't miss?",
    "Summarize the causes of the French Revolution in three paragraphs.",
    "What are some vegetarian options available in Chinese cuisine?",
]


def _load_user_prompts(jsonl_path: Path, limit: int) -> list[str]:
    prompts: list[str] = []
    if not jsonl_path.exists():
        return prompts
    with jsonl_path.open() as f:
        for line in f:
            d = json.loads(line)
            msgs = d.get("messages", [])
            if len(msgs) < 2:
                continue
            user = msgs[1].get("content", "").strip()
            if not user:
                continue
            prompts.append(user)
            if len(prompts) >= limit:
                break
    return prompts


def in_domain_prompts(limit_from_test: int = 4) -> list[str]:
    """Curated role-play prompts + first N real test-set prompts."""
    from_test = _load_user_prompts(DATA_DIR / "jack_sparrow_test.jsonl", limit_from_test)
    return CURATED_JACK_SPARROW_PROMPTS + from_test


def out_of_domain_prompts(limit_from_test: int = 4) -> list[str]:
    from_test = _load_user_prompts(DATA_DIR / "ultrachat_test.jsonl", limit_from_test)
    return CURATED_ULTRACHAT_PROMPTS + from_test


DOMAINS = {
    "Jack Sparrow (in-domain)": {
        "system": JACK_SPARROW_SYSTEM,
        "prompts": in_domain_prompts,
    },
    "General (out-of-domain)": {
        "system": ULTRACHAT_SYSTEM,
        "prompts": out_of_domain_prompts,
    },
}


if __name__ == "__main__":
    for name, info in DOMAINS.items():
        print(f"=== {name} ===")
        print(f"system: {info['system'][:80]}...")
        for i, p in enumerate(info["prompts"]()):
            print(f"  [{i}] {p[:100]}")
