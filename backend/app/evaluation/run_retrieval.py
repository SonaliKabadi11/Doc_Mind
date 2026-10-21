"""Run retrieval experiments over the evaluation dataset.

Usage (from backend/):
    python -m app.evaluation.run_retrieval --chunk-sizes 500 1000 1500 --overlaps 0 200 --detail

Documents are expected at <docs>/<doc>.pdf, where <doc> is the "doc" field in the questions.
Tune on --split dev (default). Use --split test only for final numbers.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import time
from pathlib import Path
import sys

from app.evaluation.dataset import describe, load_questions, select_split
from app.evaluation.retrieval_eval import (
    Config,
    format_breakdown,
    format_misses,
    format_table,
    run_experiment,
)


def resolve_docs(docs_dir: Path, doc_names: set[str]) -> dict[str, Path]:
    docs = {name: docs_dir / f"{name}.pdf" for name in sorted(doc_names)}
    missing = [str(p) for p in docs.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing PDF files:\n  " + "\n  ".join(missing))
    return docs


def build_grid(model: str, sizes: list[int], overlaps: list[int]) -> list[Config]:
    return [Config(model, s, o) for s in sizes for o in overlaps if o < s]


class _Tee:
    """Write console output to both the terminal and the run log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _run_logged(args: argparse.Namespace, log_path: Path) -> None:
    from app.logger import configure_logging
    configure_logging(log_path, console=False)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        output = _Tee(sys.stdout, log_file)
        error = _Tee(sys.stderr, log_file)
        with redirect_stdout(output), redirect_stderr(error):
            print(f"Retrieval evaluation started: {datetime.now().astimezone().isoformat()}")
            print(f"Full log: {log_path.resolve()}\n")
            _run(args)
            print(f"\nRetrieval evaluation finished: {datetime.now().astimezone().isoformat()}")


def _run(args: argparse.Namespace) -> None:
    from app.config.config import get_settings
    from app.ingestion.base import Ingestion  # heavy import (torch), kept out of module load

    model = args.model or get_settings().embedding_model
    all_questions = load_questions(args.questions)
    print(describe(all_questions))
    questions = select_split(all_questions, args.split)
    print(f"\nUsing split={args.split}: {len(questions)} questions")
    if args.split == "test":
        print("!! TEST SPLIT: use for final reporting only, never for choosing settings.")
    docs = resolve_docs(args.docs, {q.doc for q in questions})

    def make_ingestor(cfg: Config):
        return Ingestion(model_name=cfg.model, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    experiments = []
    for cfg in build_grid(model, args.chunk_sizes, args.overlaps):
        start = time.perf_counter()
        experiments.append(run_experiment(cfg, make_ingestor, docs, questions))
        print(f"  ran {cfg.label} in {time.perf_counter() - start:.0f}s")

    print("\n" + format_table(experiments))
    if args.detail:
        for exp in experiments:
            print("\n" + format_breakdown(exp) + "\n" + format_misses(exp))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--docs", type=Path, default=Path("eval/docs"))
    parser.add_argument("--questions", type=Path, default=Path("eval/questions"))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--model", default=None, help="embedding model (default: from settings)")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[500, 1000, 1500])
    parser.add_argument("--overlaps", type=int, nargs="+", default=[0, 200])
    parser.add_argument("--detail", action="store_true", help="per-config breakdowns and misses")
    parser.add_argument("--log-file", type=Path, help="log path (default: eval/runs/retrieval_<timestamp>.log)")
    args = parser.parse_args()
    log_path = args.log_file or Path("eval/runs") / f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _run_logged(args, log_path)


if __name__ == "__main__":
    main()
