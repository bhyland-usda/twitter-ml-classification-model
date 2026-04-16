pub fn build_combined_text(parent: &str, comment: &str) -> String {
    let p = parent.trim();
    let c = comment.trim();
    // Because parent_text is required, callers always pass a valid parent string.
    if p.is_empty() {
        // This should not happen in normal operation; enforce fail-fast at CSV parsing layer.
        if c.is_empty() {
            return "<COMMENT> </COMMENT>".to_string();
        }
        return format!("<COMMENT> {} </COMMENT>", c);
    }

    if c.is_empty() {
        return format!("<PARENT> {} </PARENT> <COMMENT> </COMMENT>", p);
    }

    format!("<PARENT> {} </PARENT> <COMMENT> {} </COMMENT>", p, c)
}
