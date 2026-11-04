"""Build all the data (EN + ZH) from the hand-curated files in sources/: sentences, then prompts.

    python build_data.py                                    # everything, both languages
    python build_data.py --lang zh                          # one language
    python build_data.py --only sentences                   # only the sentences (or: prompts)
    python build_data.py --only prompts/single_entity       # one group
    python build_data.py --only prompts/double_entity/who_quoted sentences/single_entity/masked
    python build_data.py --list                             # what exists
    python build_data.py --drafts                           # automatic drafts for hand correction -> drafts/
"""
import argparse
from collections import defaultdict
from pathlib import Path

from winogender import sources
from winogender.catalog import select
from winogender.drafts import DRAFTS
from winogender.languages import LANGUAGES

HERE = Path(__file__).resolve().parent
KINDS = ("sentences", "entities", "prompts")  # order in --list


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)  # overwrites: re-running never appends duplicate rows
    print(f"{len(df):5d} rows  {path.relative_to(HERE)}")


def output_path(output, code):
    kind, rest = output.path.split("/", 1)
    return HERE / kind / code / rest


def show(selected):
    """One line per sentence file, one line per prompt folder."""
    files, descriptions = defaultdict(lambda: defaultdict(int)), {}
    for output, language in selected:
        kind, rest = output.path.split("/", 1)
        key = (kind, output.folder.split("/", 1)[1] if kind == "prompts" else rest.removesuffix(".tsv"))
        files[key][language.code] += 1
        descriptions[key] = output.description
    kind_seen = None
    for (kind, name), per_language in sorted(files.items(), key=lambda item: (KINDS.index(item[0][0]), item[0][1])):
        if kind != kind_seen:
            if kind_seen is not None:
                print()
            print(f"{kind}/<lang>/")
            kind_seen = kind
        n = max(per_language.values())
        count = f"{n} file{'s' * (n > 1)}" if kind == "prompts" else ""
        print(f"  {name:38s} {'+'.join(per_language):6s} {count:8s} {descriptions[(kind, name)]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", nargs="+", choices=LANGUAGES, help="languages to build (default: all)")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="kinds, groups or files, e.g. prompts/single_entity")
    parser.add_argument("--list", action="store_true", help="list what can be built and exit")
    parser.add_argument("--drafts", action="store_true", help="write automatic drafts instead of the data")
    args = parser.parse_args()

    selected = list(select(args.only, args.lang))
    if not selected:
        parser.error(f"nothing matches {args.only}; see --list")
    if args.list:
        show(selected)
        return

    sources.check_alignment(list(LANGUAGES))

    if args.drafts:
        for code in args.lang or LANGUAGES:
            for name, build in DRAFTS.items():
                write(build(LANGUAGES[code]), HERE / "drafts" / code / f"{name}.tsv")
        return

    for output, language in selected:
        write(output.build(), output_path(output, language.code))


if __name__ == "__main__":
    main()
