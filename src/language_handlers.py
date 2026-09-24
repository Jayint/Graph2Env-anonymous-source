"""
Language-specific handlers for base image selection and environment setup.
Supports: Python, JavaScript, TypeScript, Rust, Go, Java, C#, C, C++, Ruby, PHP, Swift, Kotlin, Scala, R, Julia, Dart, Elixir, Haskell, Lua, Perl, Zig
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
import re
from typing import List, Dict, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 benchmark host
    import tomli as tomllib


@dataclass(frozen=True)
class LanguageRequirement:
    """One statically evidenced repository language/runtime requirement."""

    language: str
    version_constraint: str | None
    role: str
    evidence: tuple[str, ...]


class LanguageHandler(ABC):
    """Abstract base class for language-specific setup handlers."""
    
    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language name."""
        pass
    
    @abstractmethod
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate base Docker images for this language."""
        pass
    
    @abstractmethod
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """
        Detect if the repository uses this language based on structure and file contents.
        
        Args:
            repo_structure: String representation of repository structure
            files_content: Dictionary of file paths to their contents
            
        Returns:
            True if this language is detected
        """
        pass
    
    @abstractmethod
    def get_setup_instructions(self) -> str:
        """Get language-specific setup instructions for the agent."""
        pass


class PythonHandler(LanguageHandler):
    """Handler for Python projects."""
    
    @property
    def language(self) -> str:
        return "python"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate Python base images."""
        if platform == "linux":
            return [f"python:3.{v}" for v in range(6, 15)]  # 3.6 to 3.14
        else:
            return [f"python:3.{v}-windowsservercore-ltsc2022" for v in range(9, 15)]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect Python project by file extensions and config files."""
        # Strong indicators - Python-specific configuration files
        python_config_files = {
            'requirements.txt', 'setup.py', 'setup.cfg', 'pyproject.toml',
            'pipfile', 'poetry.lock', 'environment.yml', 'conda.yml',
            '.python-version', 'tox.ini', 'pytest.ini', 'pipfile.lock',
            'pdm.lock', 'uv.lock'
        }
        
        structure_lower = repo_structure.lower()
        paths = tuple(
            line.strip().replace("\\", "/").lower()
            for line in repo_structure.splitlines()
            if line.strip() and not line.rstrip().endswith("/")
        )
        has_python_source = any(path.endswith(".py") for path in paths)
        test_dirs = {"test", "tests", "__tests__", "testdata"}
        for path in paths:
            parts = path.split("/")
            if parts[-1] not in python_config_files:
                continue
            # A test-only requirements file can describe an external API test
            # harness without making the repository a Python project.  Require
            # actual Python source before promoting that evidence to a runtime.
            if set(parts[:-1]) & test_dirs and not has_python_source:
                continue
            return True
        
        # Only .py files is WEAK evidence - many projects have helper scripts.
        # Require both .py files AND typical Python project structure.
        if '.py' in structure_lower:
            # Check if it looks like a Python project structure
            # (has src/, tests/, or package with __init__.py pattern)
            if '__init__.py' in structure_lower:
                return True
            # Check for common Python project directories
            if 'src/' in structure_lower or 'tests/' in structure_lower:
                # Also need multiple .py files to be confident
                py_count = structure_lower.count('.py')
                if py_count >= 3:
                    return True
        
        return False
    
    def get_setup_instructions(self) -> str:
        return """### Python-Specific Instructions:
- Inspect pyproject.toml, setup.py/setup.cfg, requirements files, tox.ini, or lockfiles for runtime and test dependencies.
- Install the local package and test dependencies in the way the project declares; editable installs are common but not universally sufficient.
- Run pytest or the project-native test command after dependencies are installed.
"""


class JavaScriptHandler(LanguageHandler):
    """Handler for JavaScript/Node.js projects."""
    
    @property
    def language(self) -> str:
        return "javascript"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate Node.js base images."""
        if platform == "linux":
            return [f"node:{v}" for v in ["18", "20", "22", "24", "25"]]
        else:
            return ["karinali20011210/windows_server:ltsc2025_nvm"]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect Node.js project by package.json and .js files, but not TypeScript."""
        nodejs_indicators = [
            'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
            '.nvmrc', '.node-version'
        ]
        
        structure_lower = repo_structure.lower()
        for indicator in nodejs_indicators:
            if indicator in structure_lower:
                return True
        
        # package.json alone is also JS (but TS projects also have it,
        # TypeScriptHandler will override with higher priority)
        if 'package.json' in structure_lower and '.js' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### JavaScript/Node.js-Specific Instructions:
- Inspect package.json and lockfiles to choose npm, yarn, or pnpm.
- Install dependencies with the project-appropriate package manager.
- Use package.json scripts for build and test commands.
"""


class TypeScriptHandler(JavaScriptHandler):
    """Handler for TypeScript projects (inherits from JavaScript)."""
    
    @property
    def language(self) -> str:
        return "typescript"
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect TypeScript project by tsconfig.json and .ts files."""
        ts_indicators = ['tsconfig.json', 'tsconfig.build.json']
        
        structure_lower = repo_structure.lower()
        for indicator in ts_indicators:
            if indicator in structure_lower:
                return True
        
        # Check for .ts files but exclude .d.ts (declaration files)
        if '.ts' in structure_lower:
            return True
            
        return False


class RustHandler(LanguageHandler):
    """Handler for Rust projects."""
    
    @property
    def language(self) -> str:
        return "rust"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate Rust base images."""
        if platform == "linux":
            return [f"rust:1.{v}" for v in range(70, 91)]  # 1.70 to 1.90
        else:
            return [f"karinali20011210/rust-windows:1.{v}" for v in [70, 75, 80, 85, 90]]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect Rust project by Cargo.toml."""
        rust_indicators = ['cargo.toml', 'cargo.lock', 'rust-toolchain', 'rust-toolchain.toml']
        
        structure_lower = repo_structure.lower()
        for indicator in rust_indicators:
            if indicator in structure_lower:
                return True
        
        if '.rs' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### Rust-Specific Instructions:
- Use `cargo build` to build the project
- Use `cargo test` to run tests
- Use `cargo check` for faster compilation checks
- Install system dependencies if needed (check Cargo.toml for sys crates)
- Consider using `cargo install` for binary dependencies
- Check rust-toolchain or rust-toolchain.toml for specific Rust version
"""


class GoHandler(LanguageHandler):
    """Handler for Go projects."""
    
    @property
    def language(self) -> str:
        return "go"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate Go base images."""
        if platform == "linux":
            return [f"golang:1.{v}" for v in ["19", "20", "21", "22", "23", "24", "25"]]
        else:
            return [f"golang:1.{v}" for v in ["22.0-windowsservercore",
                                                  "23.0-windowsservercore",
                                                  "24.0-windowsservercore",
                                                  "25.0-windowsservercore"]]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect Go project by go.mod."""
        go_indicators = ['go.mod', 'go.sum', 'gopkg.toml', 'gopkg.lock']
        
        structure_lower = repo_structure.lower()
        for indicator in go_indicators:
            if indicator in structure_lower:
                return True
        
        if '.go' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### Go-Specific Instructions:
- Check go.mod for Go version requirement
- Use `go mod download` to download dependencies
- Use `go build ./...` to build all packages
- Use `go test ./...` to run all tests
- Use `go get` to install missing dependencies
"""


class JavaHandler(LanguageHandler):
    """Handler for Java projects."""
    
    @property
    def language(self) -> str:
        return "java"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate Java base images."""
        if platform == "linux":
            return [f"eclipse-temurin:{v}-jdk-noble" for v in ["11", "17", "21"]]
        else:
            return [f"eclipse-temurin:{v}-jdk-windowsservercore-ltsc2022" for v in ["11", "17", "21"]]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect Java project by pom.xml or build.gradle."""
        java_indicators = ['pom.xml', 'build.gradle', 'build.gradle.kts', 'gradle.properties']
        
        structure_lower = repo_structure.lower()
        for indicator in java_indicators:
            if indicator in structure_lower:
                return True
        
        if '.java' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### Java-Specific Instructions:
- Use Maven (`mvn test`) or Gradle (`gradle test`) to run tests
- Use `mvn install` or `gradle build` to build the project
- Check pom.xml (Maven) or build.gradle (Gradle) for dependencies
- Install system dependencies if needed
- Use `mvn dependency:resolve` to download dependencies
"""


class CSharpHandler(LanguageHandler):
    """Handler for C# projects."""
    
    @property
    def language(self) -> str:
        return "c#"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate C# base images."""
        if platform == "linux":
            return [f"mcr.microsoft.com/dotnet/sdk:{v}" for v in ["6.0", "7.0", "8.0", "9.0", "10.0"]]
        else:
            return [f"mcr.microsoft.com/dotnet/sdk:{v}" for v in [
                "10.0-windowsservercore-ltsc2022",
                "9.0-windowsservercore-ltsc2022",
                "8.0-windowsservercore-ltsc2022",
                "9.0-windowsservercore-ltsc2019",
                "8.0-windowsservercore-ltsc2019",
            ]]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect C# project by .csproj or .sln files."""
        cs_indicators = ['.csproj', '.sln', '.fsproj', '.vbproj']
        
        structure_lower = repo_structure.lower()
        for indicator in cs_indicators:
            if indicator in structure_lower:
                return True
        
        if '.cs' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### C#-Specific Instructions:
- Use `dotnet restore` to restore NuGet packages
- Use `dotnet build` to build the project
- Use `dotnet test` to run tests
- Check .csproj or .sln files for project configuration
- Use `dotnet run` to run the application
- Consider using `dotnet publish` for deployment builds
"""


class CppHandler(LanguageHandler):
    """Handler for C++ projects."""
    
    @property
    def language(self) -> str:
        return "c++"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate C/C++ base images."""
        if platform == "linux":
            # Use lightweight gcc images for faster pulling
            return [
                "gcc:14", "gcc:13", "gcc:12", "gcc:11",
                "buildpack-deps:jammy",  # Ubuntu 22.04 with build tools
            ]
        else:
            return [
                "mcr.microsoft.com/windows/nanoserver:ltsc2022",
                "mcr.microsoft.com/windows/servercore:ltsc2022",
            ]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect C++ project by CMakeLists.txt, .cpp, or Makefile."""
        cpp_indicators = ['cmakelists.txt', 'conanfile.txt', 'conanfile.py', 'meson.build', 'xmake.lua']
        
        structure_lower = repo_structure.lower()
        for indicator in cpp_indicators:
            if indicator in structure_lower:
                return True
        
        if '.cpp' in structure_lower or '.cc' in structure_lower or '.cxx' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### C/C++ Specific Instructions:
- Verify tools: `gcc --version ; g++ --version ; clang --version ; cmake --version ; ctest --version ; ninja --version`
- Configure with CMake:
  - `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17`
  - Use 20/17/14/11 depending on project requirement; force compiler with -DCMAKE_CXX_COMPILER=g++ if needed
- Build the project:
  - `cmake --build build --parallel`
- Run tests:
  - `ctest --test-dir build --output-on-failure`
- Dependencies: `vcpkg` or `conan` if present in the repo
- For other c/cpp repository variants not covered, decide how to build the repository yourself.
"""


class CHandler(CppHandler):
    """Handler for C projects (inherits from C++)."""
    
    @property
    def language(self) -> str:
        return "c"
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect C project."""
        c_indicators = ['cmakelists.txt', 'configure.ac', 'configure.in']
        
        structure_lower = repo_structure.lower()
        for indicator in c_indicators:
            if indicator in structure_lower:
                return True
        
        # Makefile alone is not enough (many projects use Makefile as a task runner)
        # Require .c source files (whole word match: " .c" or "/.c" or end-of-name)
        # Use regex-style check: lines ending with .c or containing /<name>.c
        import re
        if re.search(r'\b\w+\.c\b', repo_structure) and '.cpp' not in structure_lower:
            return True
            
        return False


class RubyHandler(LanguageHandler):
    """Handler for Ruby projects."""
    
    @property
    def language(self) -> str:
        return "ruby"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate Ruby base images."""
        if platform == "linux":
            return [f"ruby:{v}" for v in ["3.0", "3.1", "3.2", "3.3", "3.4"]]
        else:
            return [f"ruby:{v}" for v in ["3.2", "3.3"]]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect Ruby project by Gemfile."""
        ruby_indicators = ['gemfile', 'gemfile.lock', '.ruby-version', 'rakefile']
        
        structure_lower = repo_structure.lower()
        for indicator in ruby_indicators:
            if indicator in structure_lower:
                return True
        
        if '.rb' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### Ruby-Specific Instructions:
- Check .ruby-version for required Ruby version
- Use `bundle install` to install gem dependencies from Gemfile
- Use `bundle exec rake test` or `bundle exec rspec` to run tests
- Check Gemfile for the test framework used (rspec, minitest, etc.)
- If Rails project, run `bundle exec rails db:setup && bundle exec rails test`
"""


class PHPHandler(LanguageHandler):
    """Handler for PHP projects."""
    
    @property
    def language(self) -> str:
        return "php"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate PHP base images.
        
        Strategy: Provide PHP CLI images with version range for LLM to select based on
        project requirements. The agent will need to install git/zip/unzip if needed.
        composer images are included as options but not prioritized since they have
        fixed PHP versions that may not match project requirements.
        """
        if platform == "linux":
            # PHP CLI 镜像（主要选项）- LLM 根据项目 PHP 版本需求选择
            php_cli_images = [f"php:{v}-cli" for v in ["8.4", "8.3", "8.2", "8.1", "8.0", "7.4"]]
            # composer 镜像作为备选（已包含 git/zip/unzip，但 PHP 版本固定）
            composer_images = ["composer:2"]
            return php_cli_images + composer_images
        else:
            return [f"php:{v}" for v in ["8.4", "8.3", "8.2", "8.1"]]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect PHP project by composer.json. Distinguish from JS by .php extension."""
        structure_lower = repo_structure.lower()

        # composer.json + .php is definitive
        if 'composer.json' in structure_lower and '.php' in structure_lower:
            return True
        
        php_indicators = ['composer.lock', 'artisan', 'phpunit.xml', 'phpunit.xml.dist']
        for indicator in php_indicators:
            if indicator in structure_lower:
                return True
        
        return False
    
    def get_setup_instructions(self) -> str:
        return """### PHP-Specific Instructions:
- Check composer.json for PHP version requirement (e.g., `"php": "^8.1"`) and ensure compatibility
- Use `composer install` to install dependencies from composer.json
- Use `./vendor/bin/phpunit` to run PHPUnit tests
- Check phpunit.xml or phpunit.xml.dist for test configuration
- For Laravel: `php artisan migrate --env=testing && php artisan test`
- For Symfony: `php bin/phpunit`
"""


class KotlinHandler(LanguageHandler):
    """Handler for Kotlin projects."""
    
    @property
    def language(self) -> str:
        return "kotlin"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate Kotlin/JVM base images."""
        if platform == "linux":
            return [f"eclipse-temurin:{v}-jdk-noble" for v in ["11", "17", "21"]]
        else:
            return [f"eclipse-temurin:{v}-jdk-windowsservercore-ltsc2022" for v in ["11", "17", "21"]]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect Kotlin project by .kt files or Kotlin-specific Gradle DSL."""
        kotlin_indicators = ['build.gradle.kts', 'settings.gradle.kts']
        
        structure_lower = repo_structure.lower()
        for indicator in kotlin_indicators:
            if indicator in structure_lower:
                return True
        
        # .kt files but exclude .kts (handled above) - check raw presence
        if '.kt' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### Kotlin-Specific Instructions:
- Use Gradle (`./gradlew build`) to build the project
- Use `./gradlew test` to run tests
- Check build.gradle.kts or build.gradle for dependencies
- For Android projects, install Android SDK and use `./gradlew assembleDebug`
- For Spring Boot: `./gradlew bootRun` to run the application
"""


class ScalaHandler(LanguageHandler):
    """Handler for Scala projects."""
    
    @property
    def language(self) -> str:
        return "scala"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate Scala/JVM base images."""
        if platform == "linux":
            return [f"eclipse-temurin:{v}-jdk-noble" for v in ["11", "17", "21"]]
        else:
            return [f"eclipse-temurin:{v}-jdk-windowsservercore-ltsc2022" for v in ["11", "17", "21"]]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect Scala project by build.sbt."""
        scala_indicators = ['build.sbt', 'project/build.properties', 'project/plugins.sbt']
        
        structure_lower = repo_structure.lower()
        for indicator in scala_indicators:
            if indicator in structure_lower:
                return True
        
        if '.scala' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### Scala-Specific Instructions:
- Use `sbt compile` to compile the project
- Use `sbt test` to run tests
- Use `sbt run` to run the application
- Check build.sbt for dependencies and Scala version
- For Maven Scala projects: `mvn scala:compile && mvn test`
"""


class RHandler(LanguageHandler):
    """Handler for R projects."""
    
    @property
    def language(self) -> str:
        return "r"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate R base images."""
        if platform == "linux":
            return [f"r-base:{v}" for v in ["4.2.0", "4.3.0", "4.4.0"]]
        else:
            return ["r-base:4.3.0"]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect R project by DESCRIPTION or .R files."""
        r_indicators = ['description', 'namespace', 'renv.lock', '.rprofile']
        
        structure_lower = repo_structure.lower()
        for indicator in r_indicators:
            if indicator in structure_lower:
                return True
        
        if '.r' in structure_lower.split() or '.rmd' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### R-Specific Instructions:
- Use `Rscript -e 'install.packages(c(...))'` to install CRAN packages
- For renv projects: `Rscript -e 'renv::restore()'`
- Use `R CMD build .` to build an R package
- Use `R CMD check .` to check the package
- Use `Rscript -e 'devtools::test()'` or `Rscript -e 'testthat::test_dir("tests")'` to run tests
"""


class DartHandler(LanguageHandler):
    """Handler for Dart/Flutter projects."""
    
    @property
    def language(self) -> str:
        return "dart"
    
    def base_images(self, platform: str = "linux") -> List[str]:
        """Return candidate Dart base images."""
        if platform == "linux":
            return [f"dart:{v}" for v in ["3.0", "3.1", "3.2", "3.3", "3.4", "3.5"]]
        else:
            return ["dart:latest"]
    
    def detect_language(self, repo_structure: str, files_content: Dict[str, str]) -> bool:
        """Detect Dart/Flutter project by pubspec.yaml."""
        dart_indicators = ['pubspec.yaml', 'pubspec.lock', '.dart_tool']
        
        structure_lower = repo_structure.lower()
        for indicator in dart_indicators:
            if indicator in structure_lower:
                return True
        
        if '.dart' in structure_lower:
            return True
            
        return False
    
    def get_setup_instructions(self) -> str:
        return """### Dart/Flutter-Specific Instructions:
- Use `dart pub get` (or `flutter pub get` for Flutter) to fetch dependencies
- Use `dart test` to run tests for pure Dart projects
- Use `flutter test` to run tests for Flutter projects
- Check pubspec.yaml for SDK version constraints
- Use `dart compile exe` to compile a Dart executable
"""


# Registry of all available language handlers
LANGUAGE_HANDLERS: Dict[str, LanguageHandler] = {
    "python":     PythonHandler(),
    "javascript": JavaScriptHandler(),
    "typescript": TypeScriptHandler(),
    "rust":       RustHandler(),
    "go":         GoHandler(),
    "java":       JavaHandler(),
    "c#":         CSharpHandler(),
    "c++":        CppHandler(),
    "c":          CHandler(),
    "ruby":       RubyHandler(),
    "php":        PHPHandler(),
    "kotlin":     KotlinHandler(),
    "scala":      ScalaHandler(),
    "r":          RHandler(),
    "dart":       DartHandler(),
}


def get_language_handler(language: str) -> LanguageHandler:
    """Get the handler for a specific language."""
    if language not in LANGUAGE_HANDLERS:
        raise ValueError(f"Language '{language}' is not supported. "
                        f"Available: {list(LANGUAGE_HANDLERS.keys())}")
    return LANGUAGE_HANDLERS[language]


_DETECTION_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "target",
    "dist", "build", "__pycache__", ".tox", ".gradle", ".idea",
})
_RELEVANT_NAMES = frozenset({
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "pytest.ini", "tox.ini", ".python-version", "package.json",
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", ".nvmrc",
    ".node-version", "cargo.toml", "cargo.lock", "rust-toolchain",
    "rust-toolchain.toml", "go.mod", "go.sum", "go.work", "pom.xml",
    "build.gradle", "build.gradle.kts", "gradle.properties", "dockerfile",
})
_SOURCE_SUFFIXES = {
    "python": (".py",),
    "javascript": (".js", ".jsx"),
    "typescript": (".ts", ".tsx"),
    "rust": (".rs",),
    "go": (".go",),
    "java": (".java",),
}


def _read_detection_inventory(repo_path: str) -> tuple[str, dict[str, str]]:
    """Return a bounded repository structure and environment-relevant contents."""
    structure: list[str] = []
    contents: dict[str, str] = {}
    root_path = os.path.abspath(repo_path)
    for root, dirs, files in os.walk(root_path):
        dirs[:] = sorted(d for d in dirs if d not in _DETECTION_SKIP_DIRS)
        rel_root = os.path.relpath(root, root_path)
        for filename in sorted(files):
            rel = filename if rel_root == "." else os.path.join(rel_root, filename)
            rel = rel.replace(os.sep, "/")
            structure.append(rel)
            lower = filename.lower()
            if (
                lower in _RELEVANT_NAMES
                or rel.startswith(".github/workflows/")
                or lower.startswith("dockerfile")
            ):
                try:
                    with open(os.path.join(root, filename), encoding="utf-8") as fh:
                        contents[rel] = fh.read(128_000)
                except (OSError, UnicodeError):
                    pass
    return "\n".join(structure), contents


def _evidence_for(language: str, paths: tuple[str, ...]) -> tuple[str, ...]:
    names = {
        "python": {
            "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
            "pytest.ini", "tox.ini", ".python-version",
        },
        "javascript": {
            "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
            ".nvmrc", ".node-version",
        },
        "typescript": {
            "package.json", "tsconfig.json", "package-lock.json", "pnpm-lock.yaml",
            "yarn.lock", ".nvmrc", ".node-version",
        },
        "rust": {
            "cargo.toml", "cargo.lock", "rust-toolchain", "rust-toolchain.toml",
        },
        "go": {"go.mod", "go.sum", "go.work"},
        "java": {
            "pom.xml", "build.gradle", "build.gradle.kts", "gradle.properties",
        },
    }.get(language, set())
    suffixes = _SOURCE_SUFFIXES.get(language, ())
    strong = [
        path for path in paths
        if os.path.basename(path).lower() in names
    ]
    if not strong and suffixes:
        strong = [path for path in paths if path.lower().endswith(suffixes)][:3]
    return tuple(strong[:12])


def _first_content(files_content: dict[str, str], basename: str) -> tuple[str, str] | None:
    for path, content in sorted(files_content.items()):
        if os.path.basename(path).lower() == basename.lower():
            return path, content
    return None


def _python_version(files_content: dict[str, str]) -> str | None:
    item = _first_content(files_content, "pyproject.toml")
    if item:
        try:
            data = tomllib.loads(item[1])
            value = (data.get("project") or {}).get("requires-python")
            if not value:
                value = (((data.get("tool") or {}).get("poetry") or {})
                         .get("dependencies") or {}).get("python")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass
    item = _first_content(files_content, ".python-version")
    return item[1].strip().splitlines()[0] if item and item[1].strip() else None


def _node_version(files_content: dict[str, str]) -> str | None:
    item = _first_content(files_content, "package.json")
    if item:
        try:
            value = (json.loads(item[1]).get("engines") or {}).get("node")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass
    for name in (".nvmrc", ".node-version"):
        item = _first_content(files_content, name)
        if item and item[1].strip():
            return item[1].strip().splitlines()[0]
    return None


def _rust_version(files_content: dict[str, str]) -> str | None:
    item = _first_content(files_content, "rust-toolchain.toml")
    if item:
        try:
            value = (tomllib.loads(item[1]).get("toolchain") or {}).get("channel")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass
    item = _first_content(files_content, "rust-toolchain")
    if item and item[1].strip():
        return item[1].strip().splitlines()[0]
    item = _first_content(files_content, "cargo.toml")
    if item:
        try:
            parsed = tomllib.loads(item[1])
            candidates = (
                (parsed.get("package") or {}).get("rust-version"),
                ((parsed.get("workspace") or {}).get("package") or {}).get(
                    "rust-version"
                ),
            )
            for value in candidates:
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except Exception:
            pass
    return None


def _go_version(files_content: dict[str, str]) -> str | None:
    item = _first_content(files_content, "go.mod")
    if not item:
        return None
    match = re.search(r"(?m)^\s*go\s+([0-9]+(?:\.[0-9]+){1,2})\s*$", item[1])
    return match.group(1) if match else None


def _java_version(files_content: dict[str, str]) -> str | None:
    item = _first_content(files_content, "pom.xml")
    if item:
        text = item[1]
        enforcer = re.search(
            r"<requireJavaVersion\b[^>]*>.*?<version>\s*([^<]+)"
            r"\s*</version>.*?</requireJavaVersion>",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if enforcer:
            major = re.search(r"\d+(?:\.\d+)*", enforcer.group(1))
            if major:
                return major.group(0)
        match = re.search(r"<java\.version>\s*([^<]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    for path, content in files_content.items():
        if ".github/workflows/" not in path.replace("\\", "/").lower():
            continue
        match = re.search(
            r"(?im)^\s*java-version\s*:\s*['\"]?([0-9]+(?:\.[0-9]+)*)",
            content,
        )
        if match:
            return match.group(1)
    if item:
        for pattern in (
            r"<maven\.compiler\.release>\s*([^<]+)",
            r"<maven\.compiler\.source>\s*([^<]+)",
        ):
            match = re.search(pattern, item[1], re.IGNORECASE)
            if match:
                return match.group(1).strip()
    for name in ("build.gradle", "build.gradle.kts"):
        item = _first_content(files_content, name)
        if not item:
            continue
        match = re.search(
            r"(?:JavaLanguageVersion\.of|languageVersion\s*=)\s*\(?\s*([0-9]+)",
            item[1],
        )
        if match:
            return match.group(1)
    return None


def _language_version(language: str, files_content: dict[str, str]) -> str | None:
    if language == "python":
        return _python_version(files_content)
    if language in {"javascript", "typescript"}:
        return _node_version(files_content)
    if language == "rust":
        return _rust_version(files_content)
    if language == "go":
        return _go_version(files_content)
    if language in {"java", "kotlin"}:
        return _java_version(files_content)
    return None


def _role_scores(
    detected: tuple[str, ...],
    paths: tuple[str, ...],
    files_content: dict[str, str],
) -> dict[str, int]:
    """Score the language that owns the repository's main test interface."""
    root_names = {path.lower() for path in paths if "/" not in path}
    all_text = "\n".join(files_content.values()).lower()
    scores = {language: 0 for language in detected}
    boosts = {
        "python": (("pyproject.toml", 5), ("pytest.ini", 5), ("tox.ini", 3)),
        "javascript": (("package.json", 4),),
        "typescript": (("package.json", 4), ("tsconfig.json", 3)),
        "rust": (("cargo.toml", 5),),
        "go": (("go.mod", 5),),
        "java": (("pom.xml", 5), ("build.gradle", 5), ("build.gradle.kts", 5)),
    }
    for language in detected:
        for filename, score in boosts.get(language, ()):
            if filename in root_names:
                scores[language] += score
    command_signals = {
        "python": ("pytest", "python -m unittest"),
        "javascript": ("npm test", "pnpm test", "yarn test"),
        "typescript": ("npm test", "pnpm test", "yarn test"),
        "rust": ("cargo test",),
        "go": ("go test",),
        "java": ("mvn test", "gradlew test", "gradle test"),
    }
    for language, signals in command_signals.items():
        if language in scores and any(signal in all_text for signal in signals):
            scores[language] += 4
    return scores


def detect_languages(
    repo_structure_or_path: str,
    files_content: Dict[str, str] | None = None,
) -> tuple[LanguageRequirement, ...]:
    """Detect all repository languages with versions, roles, and evidence.

    Passing a directory performs a deterministic bounded scan.  Passing the
    historical ``repo_structure`` plus ``files_content`` form keeps the image
    selector and older callers source-compatible.
    """
    if files_content is None and os.path.isdir(repo_structure_or_path):
        repo_structure, contents = _read_detection_inventory(repo_structure_or_path)
    else:
        repo_structure = repo_structure_or_path
        contents = dict(files_content or {})

    paths = tuple(
        line.strip().replace("\\", "/")
        for line in repo_structure.splitlines()
        if line.strip() and not line.rstrip().endswith("/")
    )
    detected = tuple(
        name for name, handler in LANGUAGE_HANDLERS.items()
        if handler.detect_language(repo_structure, contents)
    )
    # TypeScript is a Node ecosystem specialization, not an additional runtime.
    # Reporting both would create duplicate Node workspaces and install blocks.
    if "typescript" in detected and "javascript" in detected:
        detected = tuple(item for item in detected if item != "javascript")
    if not detected:
        return ()

    scores = _role_scores(detected, paths, contents)
    primary = max(
        detected,
        key=lambda language: (
            scores.get(language, 0),
            language == "python",
            language in {"java", "go", "rust", "typescript", "javascript"},
        ),
    )

    # Rust commonly participates only as a Python native-extension compiler.
    python_native_rust = (
        "python" in detected
        and "rust" in detected
        and any(token in "\n".join(contents.values()).lower()
                for token in ("maturin", "setuptools-rust", "pyo3"))
    )

    requirements: list[LanguageRequirement] = []
    for language in detected:
        if language == primary:
            role = "primary_runtime"
        elif language == "rust" and python_native_rust:
            role = "build_tool"
        else:
            role = "secondary_runtime"
        requirements.append(LanguageRequirement(
            language=language,
            version_constraint=_language_version(language, contents),
            role=role,
            evidence=_evidence_for(language, paths),
        ))
    role_rank = {"primary_runtime": 0, "secondary_runtime": 1, "build_tool": 2}
    return tuple(sorted(
        requirements,
        key=lambda item: (role_rank[item.role], item.language),
    ))


def detect_language(repo_structure: str, files_content: Dict[str, str]) -> Optional[str]:
    """
    Auto-detect the primary language of the repository.
    
    Returns:
        The detected language name, or None if no language is detected.
    """
    requirements = detect_languages(repo_structure, files_content)
    if not requirements:
        return None
    primary = next(
        (item.language for item in requirements if item.role == "primary_runtime"),
        requirements[0].language,
    )
    return primary
