use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::Result;
use ignore::WalkBuilder;
use regex::Regex;

/// Extracts high-signal search terms from the raw prompt.
pub fn extract_search_terms(prompt: &str) -> HashSet<String> {
    let mut terms = HashSet::new();

    // Regex for file paths (e.g. some/path/to/file.py or foo.txt)
    let re_paths = Regex::new(r"[\w/\\.-]+\.\w+").unwrap();
    for cap in re_paths.captures_iter(prompt) {
        if let Some(m) = cap.get(0) {
            terms.insert(m.as_str().to_string());
        }
    }

    // Regex for CamelCase / PascalCase
    let re_camel = Regex::new(r"\b([A-Z]+[a-z]+[A-Z]\w*|[a-z]+[A-Z]\w*)\b").unwrap();
    for cap in re_camel.captures_iter(prompt) {
        if let Some(m) = cap.get(0) {
            terms.insert(m.as_str().to_string());
        }
    }

    // Regex for snake_case (require at least one underscore)
    let re_snake = Regex::new(r"\b_?[a-zA-Z0-9]+_[a-zA-Z0-9_]+\b").unwrap();
    for cap in re_snake.captures_iter(prompt) {
        if let Some(m) = cap.get(0) {
            terms.insert(m.as_str().to_string());
        }
    }

    // Regex for array accesses or functions
    let re_code = Regex::new(r"\b(\w+)\[.*?\]").unwrap();
    for cap in re_code.captures_iter(prompt) {
        if let Some(m) = cap.get(1) {
            terms.insert(m.as_str().to_string());
        }
    }

    terms
}

/// Represents a preprocessed context match
pub struct ContextMatch {
    pub file_path: PathBuf,
    pub snippet: String,
    pub score: usize,
}

/// Runs a heuristic search over the given directory using the extracted terms.
pub fn build_context_from_terms(dir: &Path, terms: &HashSet<String>) -> Result<Vec<ContextMatch>> {
    if terms.is_empty() {
        return Ok(Vec::new());
    }

    let mut matches_by_file: HashMap<PathBuf, Vec<(usize, String)>> = HashMap::new();

    let escaped_terms: Vec<String> = terms.iter().map(|t| regex::escape(t)).collect();
    let combined_pattern = format!("({})", escaped_terms.join("|"));
    let re = Regex::new(&combined_pattern)?;

    let walker = WalkBuilder::new(dir).hidden(true).git_ignore(true).build();

    for result in walker {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.file_type().is_some_and(|ft| ft.is_file()) {
            let path = entry.path();
            if let Ok(content) = fs::read_to_string(path) {
                for (line_num, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        matches_by_file
                            .entry(path.to_path_buf())
                            .or_default()
                            .push((line_num, line.to_string()));
                    }
                }
            }
        }
    }

    let mut context_matches = Vec::new();

    for (path, mut hits) in matches_by_file {
        hits.sort_by_key(|(ln, _)| *ln);
        let score = hits.len();

        let mut snippet = String::new();
        let mut last_line: Option<usize> = None;
        for (line_num, line) in &hits {
            if let Some(ll) = last_line
                && *line_num > ll + 1
            {
                snippet.push_str("...\n");
            }
            snippet.push_str(&format!("{}: {}\n", line_num + 1, line));
            last_line = Some(*line_num);
        }

        context_matches.push(ContextMatch { file_path: path, snippet, score });
    }

    context_matches.sort_by_key(|b| std::cmp::Reverse(b.score));
    context_matches.truncate(5);

    Ok(context_matches)
}

/// Scans the repo for structural metadata: top-level layout and test files.
pub fn scan_repo_structure(dir: &Path) -> String {
    let mut result = String::new();
    result.push_str("## Repository Structure\n");

    // Top-level directory listing (1 level deep, dirs first)
    if let Ok(entries) = std::fs::read_dir(dir) {
        result.push_str("```\n");
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            let prefix = if path.is_dir() { "  [dir] " } else { "  [file] " };
            result.push_str(&format!("{}{}\n", prefix, name));
        }
        result.push_str("```\n\n");
    }

    // Find test files (fast fd-style scan, 1 level deep)
    let test_patterns = ["test_", "_test.", "tests/", "spec_", "_spec."];
    let mut test_files: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_test = test_patterns.iter().any(|p| name.contains(p));
            if is_test {
                test_files.push(name);
            }
        }
    }
    // Also check for tests/ subdirectory
    let tests_dir = dir.join("tests");
    if tests_dir.is_dir()
        && let Ok(entries) = std::fs::read_dir(&tests_dir)
    {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".py") || name.ends_with(".rs") || name.ends_with(".js") {
                test_files.push(format!("tests/{}", name));
            }
        }
    }
    if !test_files.is_empty() {
        test_files.truncate(20);
        result.push_str("## Test Files\n");
        for tf in &test_files {
            result.push_str(&format!("  - {}\n", tf));
        }
        result.push('\n');
    }

    result
}

/// Build repository context by extracting search terms, grepping for matches,
/// and returning a context block. Results are cached by prompt hash.
pub fn preprocess_prompt(prompt: &str, dir: &Path) -> Result<Option<String>> {
    if let Some(cached) = read_preprocessor_cache(prompt, dir) {
        return Ok(Some(cached));
    }
    let terms = extract_search_terms(prompt);
    let matches = build_context_from_terms(dir, &terms)?;

    let mut result = String::new();
    result.push_str("<repository_context>\n");

    // Include repo structure for file path discovery
    result.push_str(&scan_repo_structure(dir));

    if !terms.is_empty() && !matches.is_empty() {
        result.push_str("## Relevant Code Snippets\n");
        result.push_str("(Based on terms extracted from the issue)\n\n");
        for m in matches {
            result.push_str(&format!("File: {}\n", m.file_path.display()));
            result.push_str(&m.snippet);
            result.push('\n');
        }
    }
    result.push_str("</repository_context>\n");
    write_preprocessor_cache(prompt, dir, &result);
    Ok(Some(result))
}

/// Writes preprocessor results to `.shimmer_context` in the working directory
/// so the model can read it on demand with `cat` instead of overflowing the prompt.
pub fn write_context_file(prompt: &str, dir: &Path) -> Result<Option<String>> {
    let context = match preprocess_prompt(prompt, dir)? {
        Some(ctx) => ctx,
        None => return Ok(None),
    };
    let ctx_path = dir.join(".shimmer_context");
    // Strip XML wrapper tags so the output reads naturally in cat
    let clean =
        context.replace("<repository_context>\n", "").replace("</repository_context>\n", "");
    std::fs::write(&ctx_path, &clean)?;
    Ok(Some(
        "Hint: I pre-searched the repo for relevant terms and saved results to .shimmer_context. \
         Use `cat .shimmer_context` to view them if helpful.\n"
            .to_string(),
    ))
}

fn cache_key(prompt: &str, dir: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(dir)
        .output()
        .ok()?;
    let head = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if head.is_empty() {
        return None;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    head.hash(&mut hasher);
    prompt.hash(&mut hasher);
    let hash_val = hasher.finish();
    Some(format!("{:x}_{}", hash_val, head.chars().take(8).collect::<String>()))
}

fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SHIMMER_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from("/tmp/shimmer_preprocessor_cache")
}

fn read_preprocessor_cache(prompt: &str, dir: &Path) -> Option<String> {
    let key = cache_key(prompt, dir)?;
    let path = cache_dir().join(&key);
    if path.exists() { std::fs::read_to_string(&path).ok() } else { None }
}

fn write_preprocessor_cache(prompt: &str, dir: &Path, content: &str) {
    if let Some(key) = cache_key(prompt, dir) {
        let dir_path = cache_dir();
        let _ = std::fs::create_dir_all(&dir_path);
        let _ = std::fs::write(dir_path.join(&key), content);
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_extract_search_terms() {
        let prompt = "Error in astropy/nddata/mixins/ndarithmetic.py with NDArithmeticMixin and \
                      _arithmetic_mask handling idx_customers_map[i]";
        let terms = extract_search_terms(prompt);
        assert!(terms.contains("astropy/nddata/mixins/ndarithmetic.py"));
        assert!(terms.contains("NDArithmeticMixin"));
        assert!(terms.contains("_arithmetic_mask"));
        assert!(terms.contains("idx_customers_map"));
    }

    #[test]
    fn test_build_context() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.py");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "def foo():").unwrap();
        writeln!(file, "    _arithmetic_mask = 1").unwrap();
        writeln!(file, "    return _arithmetic_mask").unwrap();

        let mut terms = HashSet::new();
        terms.insert("_arithmetic_mask".to_string());

        let matches = build_context_from_terms(dir.path(), &terms).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].score, 2);
        assert!(matches[0].snippet.contains("2:     _arithmetic_mask = 1"));
        assert!(matches[0].snippet.contains("3:     return _arithmetic_mask"));
    }

    #[test]
    fn test_preprocess_prompt() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.py");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "def _arithmetic_mask(): pass").unwrap();

        let prompt = "Fix _arithmetic_mask";
        let ctx = preprocess_prompt(prompt, dir.path()).unwrap();
        assert!(ctx.is_some());
        let ctx_str = ctx.unwrap();
    }

    #[test]
    fn test_extract_empty_prompt() {
        let terms = extract_search_terms("");
        assert!(terms.is_empty());
    }

    #[test]
    fn test_extract_no_matches() {
        let terms = extract_search_terms("hello world this is a test");
        assert!(terms.is_empty());
    }

    #[test]
    fn test_extract_array_access() {
        let terms = extract_search_terms("Fix the foo[bar] and baz[0] methods");
        assert!(terms.contains("foo"));
        assert!(terms.contains("baz"));
    }

    #[test]
    fn test_extract_duplicate_terms() {
        let terms = extract_search_terms("foo.py foo.py foo.py");
        assert_eq!(terms.len(), 1);
        assert!(terms.contains("foo.py"));
    }
}
