# VectorLessRAG

A vector-free Retrieval-Augmented Generation (RAG) system for website content. This project crawls websites, builds a hierarchical page index, and enables AI-powered question answering without the need for vector embeddings.

## Overview

VectorLessRAG takes a different approach to RAG by leveraging the natural hierarchical structure of website content instead of vector similarity search. The system:

1. **Crawls** a website and converts pages to markdown
2. **Indexes** the content into a hierarchical tree structure (one node per page)
3. **Queries** the index using LLM-guided tree navigation to find relevant pages
4. **Retrieves** full markdown content from selected pages
5. **Generates** precise answers using the complete page content

## Features

- **AI-Powered URL Filtering**: Uses LLM to intelligently filter URLs based on commercial/sales relevance (from iteration 1 onward)
- **Hierarchical Page Index**: One node per page, mirroring URL path structure
- **LLM-Guided Page Selection**: Navigates the index tree to find relevant pages
- **Full Content Retrieval**: Answers are generated from complete markdown files, not snippets
- **Parallel Summary Generation**: Fast index building with configurable parallelism
- **No Vector Database Required**: Eliminates the need for embedding models and vector stores
- **Content Deduplication**: Detects and removes duplicate content across pages

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd VectorLessRag

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your AWS credentials
```

## Environment Variables

Create a `.env` file with the following:

```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_URL_FILTER_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
ANSWER_BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
MAX_PARALLEL_SUMMARIES=10
```

## Usage

### Full Pipeline (Crawl + Index + Query)

```bash
python orchestrator.py https://example.com "What does the company do?"
```

### Step-by-Step Execution

#### 1. Crawl a Website

```bash
# Basic crawl
python website_crawler.py https://example.com

# With options
python website_crawler.py https://example.com \
    --iterations 3 \
    --max-pages 500 \
    --output-dir crawl_output

# Disable AI URL filtering (use rule-based filtering)
python website_crawler.py https://example.com --no-ai-url-filter
```

#### 2. Generate Page Index

```bash
# With AI summaries (parallel generation)
python generate_page_index.py crawl_output/example.com_20260410_105854

# With more parallel workers for faster generation
python generate_page_index.py crawl_output/example.com_20260410_105854 --max-parallel 10

# Without summaries (faster, for testing)
python generate_page_index.py crawl_output/example.com_20260410_105854 --no-summaries

# Custom model for summaries
python generate_page_index.py crawl_output/example.com_20260410_105854 \
    --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0
```

#### 3. Query the Index

```bash
# Single query
python query_page_index.py crawl_output/example.com_20260410_105854 "What services do they offer?"

# Interactive mode
python query_page_index.py crawl_output/example.com_20260410_105854 --interactive

# Verbose mode (shows reasoning and retrieved pages)
python query_page_index.py crawl_output/example.com_20260410_105854 -v "Tell me about their products"

# Increase context limit for more detailed answers
python query_page_index.py crawl_output/example.com_20260410_105854 --max-context 50000 "What services do they offer?"
```

### Using the Orchestrator

```bash
# Crawl + Index only (no query)
python orchestrator.py https://example.com --no-query

# Query existing crawl
python orchestrator.py --crawl-dir crawl_output/example.com_20260410_105854 "Your question"

# Skip index generation (use existing page_index.json)
python orchestrator.py --crawl-dir crawl_output/example.com_20260410_105854 --skip-index "Your question"
```

## Project Structure

```
VectorLessRag/
├── orchestrator.py           # Full pipeline orchestration
├── website_crawler.py        # Website crawling with AI URL filtering
├── generate_page_index.py    # Page index generation (one node per page)
├── query_page_index.py       # Tree-based retrieval and QA
├── run_full_website_pipeline.py  # Alternative crawl runner
├── helpers.py                # Utility functions
├── prompts/
│   └── url_filtering_prompt.txt  # AI URL filtering prompt
├── crawl_output/             # Output directory for crawled data
│   └── <domain>_<timestamp>/
│       ├── pages_markdown/   # Markdown files organized by URL path
│       ├── metadata/
│       │   ├── crawl_manifest.json
│       │   └── crawl_structure.json
│       └── page_index.json   # Hierarchical page index
└── README.md
```

## How It Works

### 1. Website Crawling

The crawler performs iterative breadth-first crawling:
- Starts from the homepage (iteration 0) and discovers links
- From iteration 1 onward, uses AI to filter URLs based on commercial/sales relevance
- Converts pages to clean markdown
- Removes duplicate content across pages

### 2. Page Index Generation

The indexer builds a simple tree structure:
- **One node per page** (not per heading/section)
- URL path hierarchy (e.g., `/products/software` → nested nodes)
- Each node includes: `title`, `url`, `markdown_file`, `summary`
- **Parallel summary generation** for speed (configurable workers)

Example index structure:
```json
{
  "title": "Example Company",
  "node_id": "0000",
  "url": "https://example.com",
  "markdown_file": "pages_markdown/example.com/index.md",
  "summary": "Technology company offering software solutions",
  "nodes": [
    {
      "title": "Products",
      "node_id": "0001",
      "url": "https://example.com/products",
      "markdown_file": "pages_markdown/example.com/products.md",
      "summary": "Product catalog with pricing and features"
    }
  ]
}
```

### 3. Query Processing

Query processing uses a three-stage approach:

1. **Tree Search**: LLM analyzes the page index and selects up to 15 relevant pages based on summaries
2. **Content Retrieval**: Full markdown content is loaded from selected pages (up to `max_context` characters)
3. **Answer Generation**: Complete page content is sent to the answer model for precise responses

**Context Limit**: By default, up to 300,000 characters of page content are sent to the answer model. This is enough for most use cases. If more pages are selected than fit in this limit, you'll see "Context limit reached, stopping at N pages". Use `--max-context` to increase this limit.

**Token Limit**: All model calls (except summaries) use a max_tokens of 10,240 for comprehensive responses.

## Configuration Options

### Crawler Options

| Option | Default | Description |
|--------|---------|-------------|
| `--iterations` | 2 | Number of crawl iterations |
| `--max-pages` | 2000 | Maximum pages to crawl |
| `--no-ai-url-filter` | false | Disable AI-based URL filtering |
| `--no-commercial-filter` | false | Disable commercial relevance filter |
| `--no-clean-markdown` | false | Keep raw markdown (no cleaning) |
| `--no-dedup` | false | Disable content deduplication |

### Index Options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-summaries` | false | Skip AI summary generation |
| `--model-id` | claude-haiku | Model for summary generation |
| `--max-parallel` | 10 | Max parallel summary requests |

### Query Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tree-search-model` | claude-sonnet | Model for page selection |
| `--answer-model` | claude-haiku | Model for answer generation |
| `--max-context` | 300000 | Max chars of page content for answers |
| `--verbose` | false | Show reasoning and retrieved pages |

## URL Filtering

The AI URL filter uses a customizable prompt to determine which URLs are relevant for crawling. The prompt is located at `prompts/url_filtering_prompt.txt`.

**Note**: AI filtering only applies from iteration 1 onward. Iteration 0 (homepage) always crawls all discovered links.

**Included URLs:**
- Products, services, solutions
- Pricing, plans, ROI information
- Company overview, about pages
- Case studies, testimonials
- Contact, demo request pages

**Excluded URLs:**
- Blog posts, articles (unless product-related)
- Careers, job listings
- Legal, privacy, terms pages
- Login, dashboard portals
- Support, documentation, FAQs

## Models Used

The system uses AWS Bedrock for AI capabilities:

- **URL Filtering**: Claude Haiku 4.5 (from iteration 1)
- **Summary Generation**: Claude Haiku 4.5 (parallel)
- **Tree Search**: Claude Sonnet 4.5
- **Answer Generation**: Claude Haiku 4.5

## Troubleshooting

### "Context limit reached, stopping at N pages"

This means the selected pages exceed the `max_context` character limit (default: 300,000 chars). Solutions:
- Increase limit: `--max-context 500000`
- The LLM will work with the pages that fit; this is rarely an issue with the default 300k limit

### "Bedrock summarization disabled for model..."

The model ID format is incorrect. Use inference-profile style IDs:
- ✅ `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- ❌ `anthropic.claude-haiku-4-5-20251001-v1:0`

### Crawler only gets 1 page

Check if `www.` vs non-`www.` domain mismatch. The crawler normalizes domains automatically, but ensure your starting URL is consistent.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
