import argparse
from pathlib import Path

from dcl_simulation import UtteranceRequest
from dcl_simulation.adapters import ChatCompletionLLM, LLMConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "configs/services/llm.json")
    parser.add_argument("--persona", default="A curious conversational partner.")
    parser.add_argument("--topic", default="Favorite places to walk.")
    args = parser.parse_args()
    with ChatCompletionLLM(LLMConfig.from_file(args.config)) as llm:
        llm.connect()
        request = UtteranceRequest(args.persona, args.topic, (), "Start a conversation about the topic.")
        print(llm.generate(request))


if __name__ == "__main__":
    main()
