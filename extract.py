import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
from urllib.parse import quote


REPOSITORY = "google-gemini/gemini-cli"
USER_DIRECTORIES = {
    "admin", "cli", "extensions", "get-started", "hooks", "ide-integration",
    "reference", "resources", "tools",
}
USER_FILES = {
    "index.md", "get-started.md", "quickstart.md", "extensions.md", "hooks.md",
    "ide-integration.md", "troubleshooting.md", "checkpointing.md", "sandbox.md",
    "extension.md", "telemetry.md", "core/subagents.md", "core/remote-agents.md",
    "core/gemma-setup.md", "core/local-model-routing.md",
}
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024


def is_user_document(path):
    path = PurePosixPath(path)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.suffix in {".md", ".mdx"}
        and (str(path) in USER_FILES or path.parts[0] in USER_DIRECTORIES)
    )


def read_local_docs(root_dir):
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Documentation directory does not exist: {root}")
    documents = {}
    for path in sorted(root.rglob("*.md*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.is_file() or not is_user_document(relative):
            continue
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ValueError(f"Documentation file is too large: {relative}")
        documents[relative] = path.read_bytes().decode("utf-8")
    return documents


def read_archive_docs(fileobj):
    documents = {}
    with tarfile.open(fileobj=fileobj, mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not member.isfile() or len(path.parts) < 3 or path.parts[1] != "docs":
                continue
            relative = PurePosixPath(*path.parts[2:]).as_posix()
            if not is_user_document(relative):
                continue
            if member.size > MAX_DOCUMENT_BYTES:
                raise ValueError(f"Documentation file is too large: {relative}")
            with archive.extractfile(member) as source:
                documents[relative] = source.read().decode("utf-8")
    return documents


def fetch_official_docs(ref):
    if not re.fullmatch(r"v\d+\.\d+\.\d+|[0-9a-f]{40}", ref):
        raise ValueError("Use an explicit stable tag (vX.Y.Z) or a full commit SHA, not a moving branch.")
    try:
        result = subprocess.run(
            ["gh", "api", "--hostname", "github.com", f"repos/{REPOSITORY}/commits/{ref}", "--jq", ".sha"],
            check=True, capture_output=True, text=True, timeout=60,
        )
        commit = result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("GitHub did not return a valid commit SHA.")
        with tempfile.TemporaryFile() as archive:
            subprocess.run(
                ["gh", "api", "--hostname", "github.com", f"repos/{REPOSITORY}/tarball/{commit}"],
                check=True, stdout=archive, stderr=subprocess.PIPE, text=True, timeout=180,
            )
            archive.seek(0)
            return read_archive_docs(archive), commit
    except FileNotFoundError as error:
        raise RuntimeError("Install GitHub CLI (gh) to download docs, or use --source-dir.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"GitHub download failed: {(error.stderr or '').strip()}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("GitHub download timed out. No output was written.") from error


def markdown_sections(content, fallback_title, *, mdx=False):
    lines = content.splitlines(keepends=True)
    start = 0
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() in {"---", "..."}), None)
        if end is None:
            raise ValueError("Unterminated Markdown frontmatter.")
        start = end + 1
    headings = []
    fence = None
    for index in range(start, len(lines)):
        line = lines[index]
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
            continue
        if mdx and re.fullmatch(r"import\s+.+\s+from\s+['\"].+['\"];?\s*", line):
            lines[index] = "\n"
            continue
        heading = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$", line.rstrip("\r\n"))
        if heading:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2] or "").strip()
            headings.append((index, len(heading[1]), title))
    title = next((text for _, level, text in headings if level == 1), fallback_title)
    boundaries = [(start, 0, "")] if not headings or headings[0][0] > start else []
    boundaries.extend(headings)
    hierarchy = []
    for position, (index, level, heading) in enumerate(boundaries):
        end = boundaries[position + 1][0] if position + 1 < len(boundaries) else len(lines)
        if level:
            hierarchy = [(depth, text) for depth, text in hierarchy if depth < level]
            hierarchy.append((level, heading))
        text = "".join(lines[index:end]).strip("\r\n")
        if text.strip():
            yield {
                "page_content": text, "title": title,
                "section": " > ".join(text for _, text in hierarchy) or title,
                "line_start": index + 1, "line_end": end,
            }


def extract_documents(sources, *, version=None, commit=None, collected_at=None):
    collected_at = collected_at or datetime.now(timezone.utc).isoformat()
    documents = []
    for path, content in sorted(sources.items()):
        if not is_user_document(path):
            continue
        source = f"docs/{path}"
        document_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        fallback_title = PurePosixPath(path).stem.replace("-", " ").title()
        for section in markdown_sections(content, fallback_title, mdx=path.endswith(".mdx")):
            text = section.pop("page_content")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            source_url = None
            if commit:
                source_url = (
                    f"https://github.com/{REPOSITORY}/blob/{commit}/{quote(source)}"
                    f"#L{section['line_start']}-L{section['line_end']}"
                )
            identity = f"{commit or 'local'}:{source}:{section['line_start']}:{content_hash}"
            documents.append({
                "id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                "page_content": text,
                "metadata": {
                    "schema_version": 1, "product": "gemini-cli", "version": version,
                    "commit": commit, "source": source, "source_url": source_url,
                    "language": "en", "document_hash": document_hash,
                    "content_hash": content_hash, "collected_at": collected_at, **section,
                },
            })
    if not documents:
        raise ValueError("No non-empty user documentation was found. No output was written.")
    return documents


def write_documents(documents, output_file):
    output = Path(output_file)
    with output.open("x", encoding="utf-8") as destination:
        try:
            for document in documents:
                destination.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            output.unlink()
            raise


def extract_md_to_txt(root_dir, output_file):
    """
    Finds all .md files in a directory and its subdirectories,
    and combines their content into a single .txt file.

    Args:
        root_dir (str): The path to the root directory to search.
        output_file (str): The path to the output text file.
    """
    # A list to hold the content of all markdown files found.
    all_md_content = []
    
    print(f"Starting search in directory: {root_dir}")

    # os.walk() generates the file names in a directory tree,
    # by walking the tree either top-down or bottom-up.
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file has a .md extension.
            if filename.endswith(".md"):
                # Construct the full file path.
                file_path = os.path.join(dirpath, filename)
                print(f"Found Markdown file: {file_path}")
                
                try:
                    # Open and read the content of the markdown file.
                    with open(file_path, 'r', encoding='utf-8') as md_file:
                        content = md_file.read()
                        
                        # Add a header to distinguish content from different files.
                        header = f"\n\n--- Content from: {file_path} ---\n\n"
                        all_md_content.append(header)
                        all_md_content.append(content)
                        
                except Exception as e:
                    # Handle potential reading errors.
                    error_message = f"\n\n--- Error reading file: {file_path} --- \n{e}\n"
                    all_md_content.append(error_message)
                    print(error_message)

    # Write the combined content to the output file.
    try:
        with open(output_file, 'w', encoding='utf-8') as txt_file:
            txt_file.write("".join(all_md_content))
        print(f"\nSuccessfully combined all .md files into: {output_file}")
    except Exception as e:
        print(f"\nError writing to output file: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract user-facing Gemini CLI Markdown sections into a new JSONL dataset."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-dir", type=Path, help="Local docs directory; no official revision is assumed.")
    source.add_argument("--ref", help="Official stable tag (vX.Y.Z) or full commit SHA; requires GitHub CLI (gh).")
    parser.add_argument("--output", type=Path, default=Path("gemini_cli_documents.jsonl"), help="New JSONL file; existing files are never overwritten.")
    args = parser.parse_args(argv)
    # --- Configuration ---
    # Set the root directory to the 'docs' folder inside the cloned repository.
    # You might need to change this path depending on where you run the script from.
    source_directory = args.source_dir or Path("gemini-cli/docs")

    # Set the name for the final output text file.
    destination_file = args.output

    # Check if the source directory exists before running.
    try:
        if destination_file.exists():
            raise FileExistsError(f"Output already exists: {destination_file}. Choose a new --output path.")
        commit = None
        if args.ref:
            sources, commit = fetch_official_docs(args.ref)
        else:
            sources = read_local_docs(source_directory)
        # Run the function.
        documents = extract_documents(sources, version=args.ref, commit=commit)
        write_documents(documents, destination_file)
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
        parser.exit(1, f"Error: {error}\n")
    count = len({document["metadata"]["source"] for document in documents})
    print(f"Extracted {len(documents)} sections from {count} documents into {destination_file}")
    if commit:
        print(f"Source: {REPOSITORY} at {commit} ({args.ref})")
    else:
        print("Source: local files, without a verified official revision")


if __name__ == "__main__":
    main()
