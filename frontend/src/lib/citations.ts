/**
 * Parse evidence citation tokens out of draft body markdown.
 *
 * The drafter emits citations in two stable forms:
 *   - `[<document_id>:fields]`  → derived structured-field chunk for a document
 *   - `[<document_id>:p<page>:c<chunk_index>]` → page chunk
 *
 * We treat any `[<id>]` whose contents only use the characters our evidence
 * ids actually use (lowercased hex/digits, colons, ``p``/``c`` prefixes,
 * ``fields``) as a citation. Anything else is left untouched so accidental
 * brackets in body prose are not mistakenly chipified.
 */

export interface CitationToken {
  kind: "text" | "citation";
  text: string;
  evidenceId?: string;
}

// Match a single citation token like [abcd1234:fields] or [abcd1234:p4:c0].
// We use a non-anchored regex and split the body by it so we can interleave
// text and citation spans in render order.
const CITATION_TOKEN_RE = /\[([a-z0-9]{4,}(?::[a-z0-9_]+){0,4})\]/g;

export function tokenizeBody(body: string): CitationToken[] {
  const tokens: CitationToken[] = [];
  let cursor = 0;
  for (const match of body.matchAll(CITATION_TOKEN_RE)) {
    const matchStart = match.index ?? 0;
    if (matchStart > cursor) {
      tokens.push({ kind: "text", text: body.slice(cursor, matchStart) });
    }
    tokens.push({ kind: "citation", text: match[0], evidenceId: match[1] });
    cursor = matchStart + match[0].length;
  }
  if (cursor < body.length) {
    tokens.push({ kind: "text", text: body.slice(cursor) });
  }
  return tokens;
}

export function evidenceKind(evidenceId: string): "fields" | "page" | "unknown" {
  if (evidenceId.endsWith(":fields")) return "fields";
  if (/:p\d+:c\d+$/.test(evidenceId)) return "page";
  return "unknown";
}
