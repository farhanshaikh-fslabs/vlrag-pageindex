from urllib.parse import urlparse, unquote
import re
import json
import base64
import ast


def normalize_website_url(website_url: str) -> str:
    """Complete the website URL if it is not a full URL."""
    if not website_url.startswith("http"):
        website_url = "https://" + website_url
    return website_url


def clean_markdown_content(md_content):
    
    # Remove image links and markdown links but keep the link text
    clean_text = ""
    for line in md_content.split("\n"):
        if line.strip().startswith("!") or line.strip().startswith("["):
            pass
        else:
            clean_text += line + "\n"

    # Remove hyperlinks
    clean_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', clean_text)

    # Replace multiple newlines with a single newline
    clean_text = re.sub(r'\n\n+', '\n\n', clean_text)

    return clean_text.strip()

def get_suburls(content: str, company_url: str, max_urls: int) -> list[str]:

    # Extract links from markdown content using regex
    all_urls = re.findall(r'(https?://[^\s\)]+)', content)
    all_urls = list(set(all_urls))  # Remove duplicates

    filtered_urls = [url for url in all_urls if company_url in url and url.strip().startswith("http")]
    if len(filtered_urls) > max_urls:
        filtered_urls = filtered_urls[:max_urls]
    else:
        filtered_urls = filtered_urls

    return filtered_urls

def parse_model_json_response(model_response):
    updated_response = str(model_response).strip()

    # Remove fenced code markers if present.
    updated_response = re.sub(r"^```(?:json)?\s*", "", updated_response, flags=re.IGNORECASE)
    updated_response = re.sub(r"\s*```$", "", updated_response)

    if "</thinking>" in updated_response:
        updated_response = updated_response.split("</thinking>")[-1].strip()

    if "{" in updated_response:
        updated_response = updated_response[updated_response.index("{"):].strip()
    if "}" in updated_response:
        updated_response = updated_response[:updated_response.rindex("}")+1].strip()

    # Some model/provider responses include Python repr values like Decimal('70').
    # Convert those wrappers so the payload remains JSON-compatible.
    updated_response = re.sub(
        r"Decimal\((['\"])(-?\d+(?:\.\d+)?)\1\)",
        r"\2",
        updated_response,
    )

    # Try to parse as JSON
    try:
        return json.loads(updated_response)
    except json.JSONDecodeError:
        pass

    # Fix unescaped newlines inside JSON string values
    # This handles cases where the model outputs literal newlines in strings
    try:
        fixed_response = _fix_json_newlines(updated_response)
        return json.loads(fixed_response)
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback for Python-dict-like payloads (single quotes, etc.).
    try:
        parsed = ast.literal_eval(updated_response)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (ValueError, SyntaxError):
        pass

    raise ValueError("Model response is not a valid JSON object or array")


def _fix_json_newlines(json_str: str) -> str:
    """
    Fix unescaped newlines inside JSON string values.
    This handles cases where models output literal newlines instead of \\n.
    """
    result = []
    in_string = False
    escape_next = False
    i = 0

    while i < len(json_str):
        char = json_str[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        if in_string and char == '\n':
            result.append('\\n')
            i += 1
            continue

        if in_string and char == '\r':
            i += 1
            continue

        result.append(char)
        i += 1

    return ''.join(result)


def extract_domain_name(company_input: str) -> str:
    """Extract clean company name from URL or domain.
    Removes protocol, www, subdomains, and common domain extensions for better search results.

    Examples:
    - "https://www.example.com" -> "example"
    - "example.com" -> "example"
    - "http://subdomain.example.org" -> "subdomain.example"
    - "https://blog.example.co.uk" -> "example"
    - "Example Inc" -> "example inc"
    """
    if not company_input:
        return ""

    # Remove whitespace
    company = company_input.strip()

    # Remove protocol (http://, https://) - handle both with and without slashes
    company = re.sub(r'^https?://', '', company, flags=re.IGNORECASE)
    company = re.sub(r'^https?//', '', company, flags=re.IGNORECASE)  # Handle malformed URLs like https//:

    # Remove www.
    company = re.sub(r'^www\.', '', company, flags=re.IGNORECASE)

    # Extract domain name (remove path, query params, fragments)
    if '/' in company:
        company = company.split('/')[0]
    if '?' in company:
        company = company.split('?')[0]
    if '#' in company:
        company = company.split('#')[0]

    # Remove port numbers
    if ':' in company:
        parts = company.split(':')
        # Only remove if it looks like a port number (numeric)
        if len(parts) > 1 and parts[-1].isdigit():
            company = ':'.join(parts[:-1])

    # Check if it looks like a domain (contains dots)
    if '.' in company:
        # Simple approach: split by dots and take the first part (main domain name)
        # We've already removed www, so first part is the company name
        domain_parts = company.split('.')
        domain_parts = [part for part in domain_parts if part]  # Remove empty parts

        if domain_parts:
            # Extract main domain name
            # Examples:
            # "fissionlabs.com" -> ["fissionlabs", "com"] -> "fissionlabs" (2 parts: take first)
            # "blog.fissionlabs.com" -> ["blog", "fissionlabs", "com"] -> "fissionlabs" (3+ parts: take second)
            # "fissionlabs.co.uk" -> ["fissionlabs", "co", "uk"] -> "fissionlabs" (3 parts, but country TLD: take first)

            if len(domain_parts) == 2:
                # Simple case: "company.com" -> take first part
                company = domain_parts[0]
            elif len(domain_parts) >= 3:
                # Has subdomain or country code TLD
                # Strategy: if first part is a known subdomain, take second; otherwise take first
                first_part_lower = domain_parts[0].lower()
                common_subdomains = ['www', 'blog', 'www2', 'mail', 'ftp', 'shop', 'store', 'api', 'app', 'admin', 'portal']

                if first_part_lower in common_subdomains:
                    # Has subdomain: take the main domain (second part)
                    company = domain_parts[1]
                else:
                    # Might be country code TLD like "fissionlabs.co.uk" -> take first part
                    # Or could be "company.subdomain.com" -> still take first (it's the company)
                    company = domain_parts[0]
            else:
                company = domain_parts[0]
        else:
            company = company
    else:
        # Not a domain, might be a company name - just return as is (lowercased)
        company = company.lower()

    # Remove trailing dots and whitespace
    company = company.rstrip('.').strip()

    # Ensure lowercase
    company = company.lower()

    # If empty after processing, return original lowercased (might be a company name, not URL)
    if not company:
        return company_input.strip().lower()

    return company


# ---------------------------------------------------------------------------
# Commercial-value URL filter for crawl link prioritisation
# ---------------------------------------------------------------------------

_COMMERCIAL_INCLUDE = re.compile(
    r"("
    # Products, services, solutions, offerings
    r"product[s]?|service[s]?|solution[s]?|offering[s]?|platform[s]?|feature[s]?"
    r"|capabilit(y|ies)|tool[s]?"
    # Pricing, plans, ROI
    r"|pricing|price[s]?|plan[s]?|quote[s]?|roi|value|cost"
    # Industry verticals, use cases
    r"|industr(y|ies)|vertical[s]?|use[-_]?case[s]?|sector[s]?"
    # Company overview
    r"|about|about[-_]?us|company|who[-_]?we[-_]?are|our[-_]?story|mission|vision|overview"
    # Customers, testimonials, case studies
    r"|case[-_]?stud(y|ies)|customer[s]?|client[s]?|testimonial[s]?|success[-_]?stor(y|ies)"
    r"|portfolio|implementation[s]?|result[s]?"
    # Sales / contact / demo
    r"|contact|demo|request[-_]?demo|get[-_]?started|talk[-_]?to[-_]?us|schedule"
    r"|consultation|free[-_]?trial"
    # Partners, integrations
    r"|partner[s]?|distributor[s]?|integration[s]?|ecosystem|marketplace|alliance[s]?"
    # Commercial announcements, product updates, whitepapers, e-books, press
    r"|announcement[s]?|release[-_]?note[s]?|update[s]?|changelog"
    r"|news[-_]?and[-_]?press|press[-_]?release[s]?"
    r"|whitepaper[s]?|white[-_]?paper[s]?|e[-_]?book[s]?|ebook[s]?|guide[s]?|playbook[s]?"
    r"|webinar[s]?|report[s]?"
    # How-we-work / methodology
    r"|how[-_]?we[-_]?work|methodolog(y|ies)|approach|process"
    r")",
    re.I,
)

_COMMERCIAL_EXCLUDE = re.compile(
    r"("
    # Blog / articles / news (generic)
    r"blog[-_]?post[s]?|blog$|/blog/|article[s]?|news$|/news/"
    # Careers / jobs / HR
    r"|career[s]?|job[s]?|hiring|join[-_]?us|open[-_]?position[s]?|work[-_]?with[-_]?us"
    r"|internship[s]?|recruitment"
    # Legal / compliance
    r"|legal|privacy|cookie|terms|terms[-_]?of[-_]?service|terms[-_]?of[-_]?use"
    r"|disclaimer|compliance|gdpr|ccpa"
    # Support / docs / help
    r"|support$|/support/|documentation|help[-_]?center|faq[s]?|knowledge[-_]?base"
    r"|troubleshoot"
    # Auth portals
    r"|login|log[-_]?in|logout|sign[-_]?in|sign[-_]?up|register|dashboard"
    r"|my[-_]?account|profile|forgot[-_]?password"
    # API / technical docs
    r"|api[-_]?doc[s]?|api[-_]?reference|swagger|graphql|developer[-_]?doc[s]?"
    # Investor relations / financial filings
    r"|investor[s]?|investor[-_]?relation[s]?|annual[-_]?report"
    r"|shareholder[s]?|sec[-_]?filing[s]?"
    # CSR / charity
    r"|csr|corporate[-_]?social|charit(y|ies)|donation[s]?"
    # Pagination / filters / tags / categories (generic listing)
    r"|(?:^|/)page/\d|(?:^|/)tag(?:/|$)|(?:^|/)categor(?:y|ies)(?:/|$)|(?:^|/)author(?:/|$)|/archive[s]?"
    # Old / duplicate homepages
    r"|old[-_]?home"
    r")",
    re.I,
)


def is_commercially_relevant_url(url: str) -> bool:
    """
    Return True if the URL path looks commercially relevant for sales/product
    intelligence gathering. Used to filter discovered links before deeper crawl
    iterations so we don't waste budget on blog posts, careers, legal pages, etc.
    """
    path = (urlparse(url).path or "/").strip("/").lower()
    path_decoded = unquote(path)

    if not path_decoded or path_decoded == "/":
        return True

    if _COMMERCIAL_EXCLUDE.search(path_decoded):
        return False

    if _COMMERCIAL_INCLUDE.search(path_decoded):
        return True

    # Short paths (1-2 segments) are likely top-level nav pages — keep them
    segments = [s for s in path_decoded.split("/") if s]
    if len(segments) <= 2:
        return True

    return False


def filter_commercial_urls(urls: list[str]) -> list[str]:
    """Filter a list of URLs down to only commercially relevant ones."""
    return [u for u in urls if is_commercially_relevant_url(u)]


def decode_cursor(cursor: str | None) -> dict | None:
    if not cursor:
        return None
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def encode_cursor(last_evaluated_key: dict | None) -> str | None:
    if not last_evaluated_key:
        return None
    raw = json.dumps(last_evaluated_key).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")
