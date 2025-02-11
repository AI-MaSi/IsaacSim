import os
import ast
import re
from pathlib import Path
import argparse


class ConfigClassScanner:
    def __init__(self):
        self.collected_classes = {}

    def extract_classes(self, node, parent_name=""):
        """Recursively extracts class names, including nested ones."""
        classes = []
        if isinstance(node, ast.ClassDef):
            full_name = f"{parent_name}.{node.name}" if parent_name else node.name
            classes.append((full_name, node))
            for child in node.body:
                classes.extend(self.extract_classes(child, full_name))
        return classes

    def find_config_classes(self, file_path):
        """Parses a Python file to find all @configclass decorated classes."""
        config_classes = []
        print(f"[DEBUG] Scanning for config classes: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()

        # Step 1: Regex search for direct "@configclass" usage
        config_matches = re.findall(r"@configclass\s*\n\s*class\s+(\w+)\s*(?:\(([^)]*)\))?:", file_content)

        if config_matches:
            for class_name, base_classes in config_matches:
                base_classes = base_classes.strip() if base_classes else ""
                formatted_class = f"@configclass\nclass {class_name}({base_classes}):" if base_classes else f"@configclass\nclass {class_name}:"
                config_classes.append(formatted_class)
                print(f"[DEBUG] Found (regex) @configclass: {class_name}")

        # Step 2: AST parsing for structured detection
        try:
            tree = ast.parse(file_content)
            all_classes = self.extract_classes(tree)

            for full_class_name, node in all_classes:
                for decorator in node.decorator_list:
                    if ((isinstance(decorator, ast.Name) and decorator.id == "configclass") or
                            (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                             and decorator.func.id == "configclass")):
                        base_classes = ", ".join(base.id for base in node.bases if isinstance(base, ast.Name))
                        formatted_class = f"@configclass\nclass {full_class_name}({base_classes}):" if base_classes else f"@configclass\nclass {full_class_name}:"
                        if formatted_class not in config_classes:
                            config_classes.append(formatted_class)
                            print(f"[DEBUG] Found (AST) @configclass: {full_class_name}")

        except Exception as e:
            print(f"[ERROR] Failed to parse {file_path}: {e}")

        return config_classes

    def scan_directory(self, directory):
        """Scans a directory for Python files with config classes."""
        python_files = []
        print(f"[DEBUG] Searching for Python files in: {directory}")

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    python_files.append(full_path)

        print(f"[DEBUG] Total Python files to scan: {len(python_files)}")

        for file_path in python_files:
            config_classes = self.find_config_classes(file_path)
            if config_classes:
                rel_path = os.path.relpath(file_path, directory)
                module_path = rel_path.replace(os.sep, '.').replace('.py', '')
                self.collected_classes[module_path] = config_classes


class ExampleCollector:
    def __init__(self, blacklist_file='blacklist.txt'):
        self.collected_examples = {}
        self.blacklist_file = blacklist_file
        self.BLACKLISTED_FOLDERS = self.load_blacklisted_folders()

    def load_blacklisted_folders(self):
        """Load blacklisted folders from specified blacklist file."""
        try:
            with open(self.blacklist_file, 'r', encoding='utf-8') as f:
                # Read lines, strip whitespace, and filter out empty lines and comments
                folders = {
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith('#')
                }
                print(f"[DEBUG] Loaded {len(folders)} blacklisted folders from {self.blacklist_file}")
                return folders
        except FileNotFoundError:
            print(f"[WARNING] {self.blacklist_file} not found. Using empty blacklist.")
            return set()
        except Exception as e:
            print(f"[ERROR] Failed to load {self.blacklist_file}: {e}")
            return set()

    def should_skip_folder(self, dirpath):
        """Check if a folder should be skipped."""
        if os.path.basename(dirpath).startswith('_'):
            return True

        # Convert both paths to a common format
        path_str = str(dirpath).replace('\\', '/')
        path_parts = Path(path_str).parts

        for blacklisted in self.BLACKLISTED_FOLDERS:
            blacklist_parts = Path(blacklisted).parts

            # If blacklist parts is longer than path, it can't match
            if len(blacklist_parts) > len(path_parts):
                continue

            # Check if the path parts exactly match at any position
            for i in range(len(path_parts) - len(blacklist_parts) + 1):
                if path_parts[i:i + len(blacklist_parts)] == blacklist_parts:
                    print(f"[DEBUG] Skipping blacklisted folder: {dirpath}")
                    return True

        return False

    def should_skip_file(self, filename):
        """Check if a file should be skipped."""
        return filename.startswith('_') or filename == '__init__.py'

    def clean_content(self, content):
        """Remove standard license header and clean up content."""
        # Remove the standard license header
        header_pattern = r"# Copyright \(c\) 2022-2025, The Isaac Lab Project Developers\.\n# All rights reserved\.\n#\n# SPDX-License-Identifier: BSD-3-Clause\n+"
        content = re.sub(header_pattern, '', content).lstrip()
        return content

    def collect_examples(self, source_folder):
        """Collects Python examples from the source folder."""
        print(f"[DEBUG] Collecting examples from: {source_folder}")

        for dirpath, dirnames, filenames in os.walk(source_folder):
            if self.should_skip_folder(dirpath):
                dirnames.clear()  # Stop recursing into this directory
                continue

            for filename in filenames:
                if not filename.endswith('.py') or self.should_skip_file(filename):
                    continue

                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Clean the content
                    content = self.clean_content(content)

                    # Create module path
                    rel_path = os.path.relpath(file_path, source_folder)
                    module_path = rel_path.replace(os.sep, '.').replace('.py', '')

                    # Store the file content
                    self.collected_examples[module_path] = {
                        'content': content,
                        'path': str(file_path)
                    }
                    print(f"[DEBUG] Collected example: {module_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to process {file_path}: {e}")


def save_to_txt(examples, config_classes, output_path, target_size_kb=150):
    """
    Save data in multiple text files, splitting between complete examples.
    Args:
        examples: Dictionary of examples
        config_classes: Dictionary of config classes
        output_path: Base path for output files
        target_size_kb: Target size for each file in kilobytes (will exceed to keep examples intact)
    """
    target_bytes = target_size_kb * 1024
    base_path = output_path.replace('.txt', '')

    # First, calculate total size and split examples into chunks
    example_chunks = []
    current_chunk = []
    current_size = 0

    for module_path, data in examples.items():
        # Format the complete example
        example = f"[FILE: {module_path}]\n{data['content']}\n{'=' * 50}\n\n"
        example_size = len(example.encode('utf-8'))

        # If adding this example would exceed target size and we already have examples,
        # start a new chunk
        if current_size + example_size > target_bytes and current_chunk:
            example_chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append(example)
        current_size += example_size

    # Add remaining examples
    if current_chunk:
        example_chunks.append(current_chunk)

    # Write examples chunks to separate files
    for i, chunk in enumerate(example_chunks, 1):
        file_path = f"{base_path}_part{i}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=============== EXAMPLES ===============\n\n")
            f.write(''.join(chunk))

            # Add config classes to the last file only
            if i == len(example_chunks):
                f.write("\n=============== CONFIG CLASSES ===============\n\n")
                for module_path, classes in config_classes.items():
                    f.write(f"[MODULE: {module_path}]\n")
                    for class_def in classes:
                        f.write(f"{class_def}\n")
                    f.write("-" * 50 + "\n\n")

        # Print file size for verification
        size_kb = os.path.getsize(file_path) / 1024
        print(f"Wrote part {i} to {file_path} ({size_kb:.1f}KB)")

    print(f"\nSplit output into {len(example_chunks)} parts")


def main():
    parser = argparse.ArgumentParser(description='Scrape Python examples and config classes from IsaacLab.')
    parser.add_argument('--folders', nargs='+', default=['scripts', 'source'],
                        help='List of folder paths to process (default: scripts, source)')
    parser.add_argument('--output', default='.',
                        help='Output directory for the collected data (default: current directory)')
    parser.add_argument('--max-size', type=int, default=150,
                        help='Target size in KB for each output file (default: 150)')
    parser.add_argument('--select-list', default='blacklist.txt',
                        help='Path to the blacklist file (default: blacklist.txt)')

    args = parser.parse_args()

    # First, scan for config classes in isaaclab
    config_scanner = ConfigClassScanner()
    isaaclab_path = os.path.join("source", "isaaclab", "isaaclab")
    if os.path.exists(isaaclab_path):
        print("\nScanning for config classes...")
        config_scanner.scan_directory(isaaclab_path)
    else:
        print(f"[WARNING] IsaacLab directory not found at: {isaaclab_path}")

    # Then collect examples from specified folders, using the specified blacklist file
    example_collector = ExampleCollector(blacklist_file=args.select_list)
    print("\nCollecting examples...")
    for folder in args.folders:
        if os.path.exists(folder):
            example_collector.collect_examples(folder)
        else:
            print(f"[WARNING] Folder not found: {folder}")

    # Save data in text format
    output_file = os.path.join(args.output, 'collected_data.txt')
    save_to_txt(example_collector.collected_examples,
                config_scanner.collected_classes,
                output_file,
                args.max_size)

    # Print statistics
    print("\nStatistics:")
    print(f"Examples collected: {len(example_collector.collected_examples)}")
    print(f"Config classes found: {sum(len(classes) for classes in config_scanner.collected_classes.values())}")
    print("\nConfig classes by module:")
    for module, classes in config_scanner.collected_classes.items():
        if classes:
            print(f"  {module}: {len(classes)} classes")
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()