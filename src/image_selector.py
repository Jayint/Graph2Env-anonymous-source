"""
Intelligent Base Image Selector for Docker Environment Setup.
Inspired by RepoLaunch's approach: analyze repo structure and files to select optimal base image.
"""
import os
import re
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import json

from src.constants import DEFAULT_LLM_MODEL
from src.language_handlers import (
    LanguageRequirement,
    LanguageHandler, 
    get_language_handler, 
    detect_language,
    detect_languages,
    LANGUAGE_HANDLERS
)


# Prompt for locating potentially relevant files
LOCATE_FILES_PROMPT = """Given this repository structure:
------ BEGIN REPOSITORY STRUCTURE ------
{structure}
------ END REPOSITORY STRUCTURE ------

List the most relevant files for setting up a development environment and selecting a base Docker image, including:
0. CI/CD configuration files (.github/workflows, .travis.yml, .circleci/config.yml, appveyor.yml, etc.)
1. README files and documentation
2. Dependency configuration files (requirements.txt, package.json, Cargo.toml, go.mod, pom.xml, build.gradle, Gemfile, composer.json, pubspec.yaml, etc.)
   - IMPORTANT: Include ALL pom.xml/build.gradle files from ALL subdirectories/submodules, not just the root
   - Test dependencies in submodules may reveal architecture-specific requirements (e.g., embedded-postgres, embedded-mysql)
3. Version specification files (.python-version, .nvmrc, rust-toolchain, .ruby-version, .tool-versions, etc.)
4. Dockerfile or docker-compose files (for reference only)
5. Build configuration files (Makefile, CMakeLists.txt, build.sbt, etc.)
6. Lock files (poetry.lock, yarn.lock, pnpm-lock.yaml, Cargo.lock, etc.)

Only list files that are critical for understanding:
- Programming language and version requirements
- System dependencies and external libraries
- Build and test requirements
- Test dependencies that may have architecture-specific binaries (embedded databases, native extensions)

Format each file with its relative path (relative to project root) wrapped with tag <file></file>, one per line.
Example:
<file>README.md</file>
<file>requirements.txt</file>
<file>.github/workflows/ci.yml</file>
<file>db-scheduler/pom.xml</file>
"""


# Prompt for determining if a file is relevant
DETERMINE_RELEVANCE_PROMPT = """Given a file from the repository, determine if it is relevant for:
1. Setting up a development environment
2. Selecting an appropriate base Docker image
3. Understanding language or runtime version requirements

### File:
{file}

### Reply with the following format:
<rel>Yes</rel>

or

<rel>No</rel>

Choose either Yes or No.
Yes means this file IS relevant for environment setup and base image selection (e.g., it specifies language versions, dependencies, or build steps).
No means this file is NOT relevant (e.g., pure source code, user-facing documentation, test data, or unrelated configuration).
"""


# Prompt for detecting primary language via LLM
DETECT_LANGUAGE_PROMPT = """Based on the following repository files, identify the PRIMARY programming language of this project.

------ BEGIN REPOSITORY FILES ------
{docs}
------ END REPOSITORY FILES ------

Available languages: {available_languages}

Rules:
- Choose the language that the project's MAIN source code and tests are written in.
- If multiple languages exist, pick the one most relevant to the build/test environment (e.g., the language whose package manager or test runner is declared in the project files).
- Prioritize strong evidence: language-specific manifest files (Cargo.toml, package.json, go.mod, pom.xml, requirements.txt, Gemfile, composer.json, pubspec.yaml, etc.) and version spec files (.nvmrc, .python-version, rust-toolchain, tsconfig.json, etc.) over generic file extensions.
- Do NOT infer language from partial string matches in file names (e.g., a file named "eslint.config.js" does not make the project a C project just because it contains ".c").
- Base your decision on concrete evidence from the files provided.

CRITICAL RULES FOR MIXED-LANGUAGE PROJECTS:
- Python projects with Rust extensions (pyo3, setuptools-rust, maturin): These are PYTHON projects, not Rust projects. The Rust code is compiled as a native extension for Python. Look for Python build files (setup.py, pyproject.toml, requirements.txt, tox.ini, noxfile.py) - if present alongside Cargo.toml/pyo3, the PRIMARY language is Python.
- Python projects with C/C++ extensions: Similarly, these are Python projects with native extensions.
- Node.js projects with native addons: These are JavaScript/TypeScript projects.
- The key indicator is: which language's package manager and test runner is the PRIMARY interface? setup.py/pyproject.toml → Python, package.json → JavaScript, Cargo.toml alone → Rust.

Wrap your answer in <lang> tags, e.g.: <lang>typescript</lang>
Also provide one sentence of evidence: <evidence>Found tsconfig.json and @types/node in package.json devDependencies.</evidence>
"""


# Prompt for selecting base image
SELECT_BASE_IMAGE_PROMPT = """Based on the following repository information, recommend a suitable base Docker image.

IMPORTANT CONTEXT: This Docker image will be used to set up a DEVELOPMENT/TEST environment, NOT just for running the application. The container MUST be able to:
- Install all dependencies (including test dependencies)
- Run the project's test suite successfully
- Support the full development workflow

Therefore, TEST DEPENDENCIES are just as critical as runtime dependencies for image selection.

------ BEGIN REPOSITORY FILES ------
{docs}
------ END REPOSITORY FILES ------

Detected Language: {language}

Please recommend a suitable base Docker image. Consider:
1. The programming language and version requirements specified in the files
2. Minimum version requirements (e.g., requires-python, engines.node, rust-version in Cargo.toml, java version in pom.xml)
3. CI/CD configuration that shows which versions are tested
4. Use the most specific version that satisfies constraints (avoid 'latest' if possible)
5. VERSION SELECTION STRATEGY (CRITICAL):
   - FIRST, check setup.py/pyproject.toml classifiers or requires-python for explicitly declared Python version support
   - The `Programming Language :: Python :: X.Y` classifiers in setup.py ARE the authoritative version list for that codebase
   - If classifiers list only up to e.g. 3.5, choose the HIGHEST version from that list (e.g. python:3.5), NOT a newer one
   - CI/CD config (noxfile.py, tox.ini, .travis.yml) may reflect the CURRENT HEAD's support range, not the base_commit's range — do NOT use CI config to justify choosing a version beyond what setup.py classifiers declare
   - Only fall back to "prefer newer" when there are NO explicit version declarations at all
   - For Rust: prefer rust:1.75 or newer unless a specific older version is clearly required
   - For Node.js: prefer the LTS version (e.g., node:20) unless a lower version is explicitly required
6. For PHP projects specifically:
   - php:X.x-cli images are minimal and require installing git/zip/unzip manually for composer
   - composer:2 image includes git/zip/unzip pre-installed but has a FIXED PHP version (currently 8.5)
   - Choose composer:2 ONLY if the project has no strict PHP version requirement AND you want the convenience of pre-installed tools
   - Otherwise, choose a php:X.x-cli image matching the project's PHP requirement
7. ARCHITECTURE COMPATIBILITY (CRITICAL for test environments):
   - Java projects using embedded databases (embedded-postgres, embedded-mysql, zonky-pg-embedded, etc.) often lack ARM64 binaries
   - Native extensions (Rust crates with C bindings, Python packages with C extensions) may have ARM64 issues
   - TEST DEPENDENCIES with architecture-specific binaries WILL cause test failures on ARM64 hosts (Apple Silicon Macs)
   - If you see ANY such dependencies in the project (including test scope), you MUST add an <arch_note> tag
   - Also emit exactly one structured platform decision:
     <platform>linux/amd64</platform> only when concrete evidence requires x86 emulation;
     otherwise emit <platform>native</platform>.
   - Example: <arch_note>This project uses embedded-postgres which lacks ARM64 binaries.</arch_note>
     <platform>linux/amd64</platform>

Select a base image from the following candidate list:
{candidate_images}
Wrap the image name in a block like <image>python:3.9</image> to indicate your choice.
You MUST select an image from the candidate list above.
If there are architecture compatibility concerns, wrap them in <arch_note>...</arch_note> tags.
Always include either <platform>native</platform> or <platform>linux/amd64</platform>.
"""


class ImageSelector:
    """
    Intelligent base image selector that analyzes repository structure and content
    to recommend the optimal Docker base image.
    """
    
    # Size threshold for file content (256KB)
    FILE_SIZE_THRESHOLD = 128 * 1000 * 2
    MAX_STRUCTURE_LINES = 600
    MAX_DOCS_CHARS = 24000
    MAX_FILE_SNIPPET_CHARS = 6000
    
    def __init__(self, client: OpenAI, model: str = DEFAULT_LLM_MODEL):
        self.client = client
        self.model = model
        self._log_dir: Optional[str] = None
        self._log_counter: int = 0
        self.token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def _init_log_dir(self, log_dir: str):
        """Initialize log directory and reset counter."""
        self._log_dir = log_dir
        self._log_counter = 0
        os.makedirs(log_dir, exist_ok=True)

    def _write_llm_log(self, user_content: str, assistant_content: str, label: str = ""):
        """Write a single LLM call to {n}.md in RepoLaunch examples format."""
        if not self._log_dir:
            return
        idx = self._log_counter
        self._log_counter += 1
        header = f"[{label}] " if label else ""
        content = (
            f"##### LLM INPUT ({header}call #{idx}) #####\n"
            f"================================ Human Message =================================\n\n"
            f"{user_content}\n\n"
            f"##### LLM OUTPUT #####\n"
            f"================================== Ai Message ==================================\n\n"
            f"{assistant_content}\n"
        )
        path = os.path.join(self._log_dir, f"{idx}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _write_structure_log(self, structure: str):
        """Write the generated repo structure to structure.txt."""
        if not self._log_dir:
            return
        path = os.path.join(self._log_dir, "structure.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("------ BEGIN REPOSITORY STRUCTURE ------\n")
            f.write(structure)
            f.write("\n------ END REPOSITORY STRUCTURE ------\n")

    def _record_usage(self, response):
        usage = getattr(response, "usage", None)
        if not usage:
            return
        self.token_usage["input_tokens"] += usage.prompt_tokens
        self.token_usage["output_tokens"] += usage.completion_tokens
        self.token_usage["total_tokens"] += usage.total_tokens

    def get_token_usage(self) -> Dict[str, int]:
        return dict(self.token_usage)

    def _llm_detect_language(self, docs: str) -> Optional[str]:
        """Use LLM to detect the primary language from relevant file contents."""
        available_languages = list(LANGUAGE_HANDLERS.keys())
        prompt = DETECT_LANGUAGE_PROMPT.format(
            docs=docs[:6000],  # 限制长度，语言检测不需要太多内容
            available_languages=", ".join(available_languages)
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            self._record_usage(response)
            content = response.choices[0].message.content
            self._write_llm_log(prompt, content, label="detect_language")

            match = re.search(r'<lang>(.*?)</lang>', content)
            if match:
                lang = match.group(1).strip().lower()
                if lang in LANGUAGE_HANDLERS:
                    return lang
                # 尝试常见别名
                aliases = {"js": "javascript", "ts": "typescript", "c++": "c++", "cpp": "c++"}
                if lang in aliases and aliases[lang] in LANGUAGE_HANDLERS:
                    return aliases[lang]
        except Exception as e:
            print(f"[ImageSelector] LLM language detection failed: {e}")
        return None

    def _write_summary_log(self, potential_files: List[str], relevant_files: List[str],
                           detected_language: str, selected_image: str,
                           platform_override: Optional[str] = None,
                           detection_method: str = "unknown",
                           language_requirements: tuple[LanguageRequirement, ...] = ()):
        """Write summary.json with key results."""
        if not self._log_dir:
            return
        summary = {
            "potential_files": potential_files,
            "relevant_files": relevant_files,
            "detected_language": detected_language,
            "detected_languages": [
                {
                    "language": item.language,
                    "version_constraint": item.version_constraint,
                    "role": item.role,
                    "evidence": list(item.evidence),
                }
                for item in language_requirements
            ],
            "detection_method": detection_method,
            "selected_image": selected_image,
            "platform_override": platform_override,
            "total_llm_calls": self._log_counter,
        }
        path = os.path.join(self._log_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def select_base_image(
        self, 
        repo_path: str, 
        platform: str = "linux",
        language_hint: Optional[str] = None,
        log_dir: Optional[str] = None,
        language_requirements: tuple[LanguageRequirement, ...] | None = None,
    ) -> Tuple[str, LanguageHandler, str, Optional[str]]:
        """
        Analyze repository and select optimal base image.
        
        Args:
            repo_path: Path to the cloned repository
            platform: Target platform (linux or windows)
            language_hint: Optional language hint
            log_dir: Directory to write LLM call logs (RepoLaunch examples format)
            
        Returns:
            Tuple of (selected_image, language_handler, docs_content, platform_override)
            platform_override is "linux/amd64" if ARM64 compatibility issues detected, else None
        """
        if log_dir:
            self._init_log_dir(log_dir)

        print("[ImageSelector] Analyzing repository structure...")
        
        # Step 1: Generate repository structure
        repo_structure = self._generate_repo_structure(repo_path)
        self._write_structure_log(repo_structure)
        
        # Step 2: Locate potentially relevant files
        potential_files = self._locate_potential_files(repo_structure)
        print(f"[ImageSelector] Found {len(potential_files)} potentially relevant files")
        
        # Step 3: Determine relevance of each file
        relevant_files = self._filter_relevant_files(repo_path, potential_files)
        print(f"[ImageSelector] {len(relevant_files)} files confirmed relevant")
        
        # Step 4: Read content of relevant files
        files_content = self._read_files_content(repo_path, relevant_files)
        
        # Step 5: Build docs content (needed for both language detection and image selection)
        docs = self._build_docs_content(files_content)

        # Step 6: Detect the language set on the Host. The legacy LLM stage is
        # retained only when no structured result was supplied.
        structured_languages_supplied = language_requirements is not None
        detected_set = (
            tuple(language_requirements)
            if structured_languages_supplied
            else detect_languages(repo_path)
        )
        self.last_detected_languages = detected_set
        if language_hint:
            detected_language = language_hint
            detection_method = "hint"
        elif structured_languages_supplied and detected_set:
            detected_language = next(
                (
                    item.language for item in detected_set
                    if item.role == "primary_runtime"
                ),
                detected_set[0].language,
            )
            detection_method = "static-set"
        else:
            detected_language = self._llm_detect_language(docs)
            detection_method = "llm"
            if not detected_language:
                # Fallback to rule-based detection
                detected_language = detect_language(repo_structure, files_content)
                detection_method = "rules"
            if not detected_language:
                detected_language = "python"
                detection_method = "default"
        print(f"[ImageSelector] Detected language: {detected_language} (via {detection_method})")
        
        # Step 7: Get language handler and candidate images
        language_handler = get_language_handler(detected_language)
        candidate_images = language_handler.base_images(platform)
        
        # Step 8: Use LLM to select base image
        selected_image, platform_override = self._llm_select_base_image(
            docs, detected_language, candidate_images
        )
        
        print(f"[ImageSelector] Selected base image: {selected_image}")

        self._write_summary_log(
            potential_files,
            relevant_files,
            detected_language,
            selected_image,
            platform_override=platform_override,
            detection_method=detection_method,
            language_requirements=detected_set,
        )
        
        return selected_image, language_handler, docs, platform_override
    
    def _generate_repo_structure(self, repo_path: str) -> str:
        """Generate a text representation of repository structure."""
        structure_lines = []

        # Only skip directories with no relevance to env setup
        SKIP_DIRS = {
            '__pycache__', 'node_modules', 'target', 'build', 'dist',
            '.git', '.venv', 'venv', '.mypy_cache', '.pytest_cache',
            '.tox', '.eggs', '.idea', '.vscode', 'logs',
        }

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)

            level = root.replace(repo_path, '').count(os.sep)
            indent = '  ' * level
            folder_name = os.path.basename(root) or os.path.basename(repo_path)
            structure_lines.append(f"{indent}{folder_name}/")

            sub_indent = '  ' * (level + 1)
            for file in sorted(files):
                structure_lines.append(f"{sub_indent}{file}")

        return '\n'.join(structure_lines)
    
    def _locate_potential_files(self, repo_structure: str) -> List[str]:
        """Use LLM to identify potentially relevant files from structure."""
        structure_lines = repo_structure.split('\n')
        if len(structure_lines) > self.MAX_STRUCTURE_LINES:
            truncated_structure = '\n'.join(structure_lines[:self.MAX_STRUCTURE_LINES])
            truncated_structure += (
                f"\n... ({len(structure_lines) - self.MAX_STRUCTURE_LINES} more lines truncated)"
            )
        else:
            truncated_structure = repo_structure

        prompt = LOCATE_FILES_PROMPT.format(structure=truncated_structure)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        self._record_usage(response)
        
        content = response.choices[0].message.content
        self._write_llm_log(prompt, content, label="locate_files")
        
        # Parse <file> tags
        potential_files = []
        for line in content.split('\n'):
            line = line.strip()
            if '<file>' in line and '</file>' in line:
                match = re.search(r'<file>(.*?)</file>', line)
                if match:
                    potential_files.append(match.group(1).strip())
        
        return list(set(potential_files))  # Remove duplicates
    
    def _filter_relevant_files(self, repo_path: str, potential_files: List[str]) -> List[str]:
        """Filter files by relevance using LLM."""
        relevant_files = []
        
        for file_path in potential_files:
            full_path = os.path.join(repo_path, file_path)
            
            # Skip if doesn't exist or is directory
            if not os.path.exists(full_path) or os.path.isdir(full_path):
                continue
            
            # Skip if too large
            try:
                size = os.path.getsize(full_path)
                if size > self.FILE_SIZE_THRESHOLD:
                    continue
            except OSError:
                continue
            
            # Read file content
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(self.FILE_SIZE_THRESHOLD)
            except Exception:
                continue
            
            # Build file info for LLM
            file_info = f"""------ START FILE {file_path} ------
{content}
------ END FILE {file_path} ------"""
            
            # Ask LLM if relevant
            prompt = DETERMINE_RELEVANCE_PROMPT.format(file=file_info)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                self._record_usage(response)
                
                result = response.choices[0].message.content
                self._write_llm_log(prompt, result, label=f"relevance:{file_path}")
                is_relevant = '<rel>Yes</rel>' in result
                print(f"[ImageSelector]   {'✓' if is_relevant else '✗'} {file_path}")
                if is_relevant:
                    relevant_files.append(file_path)
            except Exception as e:
                print(f"[ImageSelector] Warning: Error checking {file_path}: {e}")
                continue
        
        return relevant_files
    
    def _read_files_content(self, repo_path: str, file_paths: List[str]) -> Dict[str, str]:
        """Read content of relevant files."""
        content_dict = {}
        
        for file_path in file_paths:
            full_path = os.path.join(repo_path, file_path)
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content_dict[file_path] = f.read(self.FILE_SIZE_THRESHOLD)
            except Exception:
                continue
        
        return content_dict
    
    # Files that explicitly declare language/version requirements — always show first
    VERSION_PRIORITY_FILES = [
        'setup.py', 'setup.cfg', 'pyproject.toml', 'requirements.txt',
        'package.json', 'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle',
        'Gemfile', 'composer.json', 'pubspec.yaml',
        '.python-version', '.nvmrc', 'rust-toolchain', '.ruby-version',
        'tox.ini', 'Makefile',
    ]

    def _build_docs_content(self, files_content: Dict[str, str]) -> str:
        """Build combined docs content from files, version-declaration files first."""
        # Sort: priority files first (by their rank in VERSION_PRIORITY_FILES), then the rest
        def sort_key(file_path):
            basename = os.path.basename(file_path)
            try:
                return self.VERSION_PRIORITY_FILES.index(basename)
            except ValueError:
                return len(self.VERSION_PRIORITY_FILES)

        sorted_files = sorted(files_content.keys(), key=sort_key)

        docs_parts = ["------ BEGIN RELEVANT FILES ------\n"]
        current_size = len(docs_parts[0])
        for file_path in sorted_files:
            content = files_content[file_path][:self.MAX_FILE_SNIPPET_CHARS]
            if len(files_content[file_path]) > self.MAX_FILE_SNIPPET_CHARS:
                content += f"\n... ({len(files_content[file_path]) - self.MAX_FILE_SNIPPET_CHARS} more characters truncated)"

            block = f"File: {file_path}\n```\n{content}\n```\n"
            if current_size + len(block) > self.MAX_DOCS_CHARS:
                docs_parts.append(
                    f"... additional relevant files truncated to stay within {self.MAX_DOCS_CHARS} characters ...\n"
                )
                break
            docs_parts.append(block)
            current_size += len(block)
        
        docs_parts.append("------ END RELEVANT FILES ------")
        return '\n'.join(docs_parts)
    
    def _llm_select_base_image(
        self, 
        docs: str, 
        language: str, 
        candidate_images: List[str]
    ) -> Tuple[str, Optional[str]]:
        """
        Use LLM to select the best base image from candidates.
        
        Returns:
            Tuple of (selected_image, platform_override)
            platform_override is "linux/amd64" if ARM64 compatibility issues detected, else None
        """
        prompt = SELECT_BASE_IMAGE_PROMPT.format(
            docs=docs,
            language=language,
            candidate_images=candidate_images
        )
        
        max_retries = 5
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )
            self._record_usage(response)
            
            content = response.choices[0].message.content
            self._write_llm_log(
                "\n".join(m["content"] for m in messages if m["role"] == "user"),
                content,
                label=f"select_image:attempt{attempt}"
            )
            
            # Extract image from <image> tag
            match = re.search(r'<image>(.*?)</image>', content)
            if match:
                selected_image = match.group(1).strip()
                if selected_image in candidate_images:
                    # Architecture notes are free-form explanation, not a
                    # machine-readable decision.  In particular, a harmless
                    # sentence such as "supports ARM64" must not force QEMU.
                    arch_match = re.search(r'<arch_note>(.*?)</arch_note>', content, re.DOTALL)
                    platform_override = None
                    if arch_match:
                        arch_note = arch_match.group(1).strip()
                        print(f"[ImageSelector] Architecture note: {arch_note}")
                    platform_match = re.search(
                        r'<platform>\s*(native|linux/amd64)\s*</platform>',
                        content,
                        re.IGNORECASE,
                    )
                    if platform_match and platform_match.group(1).lower() == "linux/amd64":
                        platform_override = "linux/amd64"
                        print(f"[ImageSelector] Suggesting platform override: {platform_override}")
                    elif not platform_match and arch_match:
                        # Backward-compatible fallback for models that omit the
                        # structured tag: accept only an explicit incompatibility
                        # claim, never the mere presence of "ARM64"/"amd64".
                        lowered = arch_note.lower()
                        incompatible = re.search(
                            r"(?:lacks?|does not support|unsupported|incompatible|unavailable)"
                            r".{0,80}(?:arm64|aarch64)|"
                            r"(?:arm64|aarch64).{0,80}"
                            r"(?:unsupported|incompatible|unavailable)",
                            lowered,
                        )
                        explicit_amd64 = re.search(
                            r"(?:requires?|must use|run on|use)\s+(?:the\s+)?linux/amd64",
                            lowered,
                        )
                        if incompatible or explicit_amd64:
                            platform_override = "linux/amd64"
                            print(f"[ImageSelector] Suggesting platform override: {platform_override}")
                    return selected_image, platform_override
                else:
                    # Image not in candidates, ask again
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"The image '{selected_image}' is not in the candidate list. "
                                   f"Please select from: {candidate_images}"
                    })
            else:
                # No <image> tag found
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "Please wrap the image name in <image> tags, e.g., <image>python:3.9</image>"
                })
        
        # Fallback: return first candidate if all retries failed
        print("[ImageSelector] Warning: Could not get valid selection, using fallback")
        return candidate_images[len(candidate_images) // 2], None  # Middle option
