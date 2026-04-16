use once_cell::sync::Lazy;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

static URL_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"https?://\S+").unwrap());
static MENTION_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"@\w+").unwrap());
static WHITESPACE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
static TOKEN_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\p{L}\p{N}_']+").unwrap());

pub fn normalize_text(input_text: &str) -> String {
    let unicode_normalized_text = input_text.nfkc().collect::<String>().to_lowercase();
    let text_with_normalized_urls = URL_PATTERN.replace_all(&unicode_normalized_text, " urltoken ");
    let text_with_normalized_mentions =
        MENTION_PATTERN.replace_all(&text_with_normalized_urls, " usertoken ");
    WHITESPACE_PATTERN
        .replace_all(&text_with_normalized_mentions, " ")
        .trim()
        .to_string()
}

pub fn tokenize(normalized_text: &str) -> Vec<String> {
    TOKEN_PATTERN
        .find_iter(normalized_text)
        .map(|token_match| token_match.as_str().to_string())
        .collect()
}
