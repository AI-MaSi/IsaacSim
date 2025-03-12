import os
import argparse
from pathlib import Path

# ============ COMMON SKIP LISTS ============
COMMON_SKIP_FOLDERS = {
    '__pycache__',
    '.pytest_cache',
    '.ipynb_checkpoints',
    '.git',
    '.vscode',
    '.idea',
    'agents',
}

SKIP_FILE_PATTERNS = {
    '.pyc',
    '.pyo',
    '.pyd',
    '.so',
    '.dll',
    '.cache',
    '.log',
    '.tmp',
    '.temp'
}


def load_blacklist(filename):
    """Load blacklisted folders from a text file."""
    try:
        with open(filename, 'r') as f:
            return {line.strip() for line in f
                    if line.strip() and not line.strip().startswith('#')}
    except FileNotFoundError:
        print(f"Warning: Blacklist file '{filename}' not found. Using empty blacklist.")
        return set()


def get_max_filename_length():
    return 255 # you could create a sys check here


def check_filename_length(filename, max_length):
    if len(filename) > max_length:
        return False, f"Warning: Generated filename '{filename}' is {len(filename)} characters long, exceeding the limit of {max_length} characters."
    return True, None


def should_skip_folder(dirname, full_path, blacklisted_folders):
    if dirname.startswith('_'):
        return True

    if dirname in COMMON_SKIP_FOLDERS or any(part in COMMON_SKIP_FOLDERS for part in Path(full_path).parts):
        return True

    path_str = str(full_path).replace('\\', '/')
    path_parts = Path(path_str).parts

    for blacklisted in blacklisted_folders:
        blacklist_parts = Path(blacklisted).parts
        if len(blacklist_parts) > len(path_parts):
            continue
        if path_parts[-len(blacklist_parts):] == blacklist_parts:
            return True
    return False


def should_skip_file(filename):
    if filename.startswith('_') or filename == 'setup.py':
        return True
    return any(filename.endswith(pattern) for pattern in SKIP_FILE_PATTERNS)


def clean_content(content):
    """Clean and minimize the content for LLM processing."""
    # Remove empty lines
    lines = [line.rstrip() for line in content.splitlines() if line.strip()]

    # Remove comments that are on their own lines
    lines = [line for line in lines if not line.lstrip().startswith('#')]

    # Remove docstrings
    in_docstring = False
    cleaned_lines = []
    for line in lines:
        if '"""' in line or "'''" in line:
            # Handle single-line docstrings
            if line.count('"""') == 2 or line.count("'''") == 2:
                continue
            # Toggle docstring state
            in_docstring = not in_docstring
            continue
        if not in_docstring:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def process_files(source_folders, output_dir, blacklisted_folders):
    """Process Python files and convert them to minimized text files."""
    processed_count = 0
    total_size_before = 0
    total_size_after = 0
    skipped_files = []
    skipped_count = {'folders': 0, 'files': 0}
    max_length = get_max_filename_length()

    output_dir.mkdir(exist_ok=True)

    for source_folder in source_folders:
        if not os.path.exists(source_folder):
            print(f"Warning: Folder '{source_folder}' does not exist, skipping...")
            continue

        for dirpath, dirnames, filenames in os.walk(source_folder):
            rel_path = os.path.relpath(dirpath, Path.cwd())

            # Filter directories
            before_count = len(dirnames)
            dirnames[:] = [d for d in dirnames if not should_skip_folder(d, Path(dirpath) / d, blacklisted_folders)]
            skipped_count['folders'] += before_count - len(dirnames)

            if should_skip_folder(os.path.basename(dirpath), rel_path, blacklisted_folders):
                skipped_count['folders'] += 1
                continue

            for filename in filenames:
                if not filename.endswith('.py') or should_skip_file(filename):
                    skipped_count['files'] += 1
                    continue

                rel_path = os.path.relpath(dirpath, source_folder)

                # Create output filename
                if rel_path == '.':
                    new_filename = filename
                else:
                    path_prefix = rel_path.replace(os.sep, '.')
                    new_filename = f"{path_prefix}.{filename}"

                if len(source_folders) > 1:
                    new_filename = f"{Path(source_folder).name}.{new_filename}"

                # Change extension to .txt
                new_filename = new_filename[:-3] + '.txt'

                # Check filename length
                is_valid, warning = check_filename_length(new_filename, max_length)
                if not is_valid:
                    skipped_files.append({
                        'original': filename,
                        'generated': new_filename,
                        'warning': warning
                    })
                    continue

                src_file = Path(dirpath) / filename
                dst_file = output_dir / new_filename

                # Process the file
                with open(src_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_size = len(content.encode('utf-8'))
                total_size_before += original_size

                cleaned_content = clean_content(content)

                with open(dst_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)

                new_size = len(cleaned_content.encode('utf-8'))
                total_size_after += new_size

                processed_count += 1
                reduction = ((original_size - new_size) / original_size) * 100
                print(f"Processed: {filename} -> {new_filename}")
                print(f"Size reduction: {reduction:.1f}% ({original_size:,} -> {new_size:,} bytes)")

    print("\nSummary:")
    print(f"Processed: {processed_count} files")
    print(f"Skipped: {skipped_count['files']} files")
    print(f"Skipped: {skipped_count['folders']} folders")
    print(f"Total size before: {total_size_before:,} bytes")
    print(f"Total size after: {total_size_after:,} bytes")
    print(f"Overall reduction: {((total_size_before - total_size_after) / total_size_before) * 100:.1f}%")

    if skipped_files:
        print("\nThe following files were skipped due to filename length limits:")
        for file in skipped_files:
            print(f"\nOriginal: {file['original']}")
            print(f"Generated: {file['generated']}")
            print(file['warning'])


def main():
    parser = argparse.ArgumentParser(
        description='Convert Python files to minimized text files for LLM processing')
    parser.add_argument(
        '--folders',
        nargs='+',
        default=['scripts', 'source'],
        help='List of folder paths to process'
    )
    parser.add_argument(
        '--use-list',
        type=str,
        help='Name of the blacklist file to use (should be a .txt file in the same directory)'
    )

    args = parser.parse_args()
    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir / "txt_output"

    # Load blacklist from file if specified
    blacklisted_folders = set()
    if args.use_list:
        blacklist_file = script_dir / args.use_list
        if not blacklist_file.name.endswith('.txt'):
            blacklist_file = blacklist_file.with_suffix('.txt')
        blacklisted_folders = load_blacklist(blacklist_file)

    print(f"Processing folders: {', '.join(args.folders)}")
    print(f"Maximum filename length: {get_max_filename_length()} characters")
    print(f"Using blacklist file: {args.use_list if args.use_list else 'None'}")
    print(f"Blacklisted folders: {', '.join(sorted(blacklisted_folders))}\n")

    process_files(args.folders, output_dir, blacklisted_folders)
    print("\nDone!")


if __name__ == "__main__":
    main()