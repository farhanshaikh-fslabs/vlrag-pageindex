# VectorLessRAG

A vector-free Retrieval-Augmented Generation (RAG) system for website content. This project crawls websites, builds a hierarchical page index, and enables AI-powered question answering without the need for vector embeddings.

## Overview

VectorLessRAG takes a different approach to RAG by leveraging the natural hierarchical structure of website content instead of vector similarity search. The system:

1. **Crawls** a website and converts pages to markdown
2. **Indexes** the content into a hierarchical tree structure based on URL paths and heading levels
3. **Queries** the index using LLM-guided tree navigation to find relevant content
4. **Generates** answers using the retrieved context

## Features

- **AI-Powered URL Filtering**: Uses LLM to intelligently filter URLs based on commercial/sales relevance
- **Hierarchical Page Index**: Mirrors URL path structure with nested heading hierarchies
- **LLM-Guided Tree Search**: Navigates the content tree to find relevant sections
- **No Vector Database Required**: Eliminates the need for embedding models and vector stores
- **Automatic Summarization**: Generates concise summaries for each node using Bedrock
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
# With AI summaries
python generate_page_index.py crawl_output/example.com_20260410_105854

# Without summaries (faster, for testing)
python generate_page_index.py crawl_output/example.com_20260410_105854 --no-summaries

# Custom model for summaries
python generate_page_index.py crawl_output/example.com_20260410_105854 \
    --model-id anthropic.claude-3-haiku-20240307-v1:0
```

#### 3. Query the Index

```bash
# Single query
python query_page_index.py crawl_output/example.com_20260410_105854 "What services do they offer?"

# Interactive mode
python query_page_index.py crawl_output/example.com_20260410_105854 --interactive

# Verbose mode (shows reasoning and context)
python query_page_index.py crawl_output/example.com_20260410_105854 -v "Tell me about their products"
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
├── generate_page_index.py    # Hierarchical index generation
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
- Starts from the homepage and discovers links
- Uses AI to filter URLs based on commercial/sales relevance
- Converts pages to clean markdown
- Removes duplicate content across pages

### 2. Page Index Generation

The indexer builds a tree structure:
- URL path hierarchy (e.g., `/products/software` → nested nodes)
- Markdown heading hierarchy within each page
- AI-generated summaries for each node
- Content previews for quick reference

Example index structure:
```json
{
  "title": "Example Company",
  "node_id": "0000",
  "url": "https://example.com",
  "summary": "Technology company offering software solutions",
  "nodes": [
    {
      "title": "Products",
      "node_id": "0001",
      "url": "https://example.com/products",
      "nodes": [...]
    }
  ]
}
```

### 3. Tree-Based Retrieval

Query processing uses LLM-guided navigation:
1. **Tree Search**: LLM analyzes the index tree and identifies relevant nodes
2. **Content Resolution**: Selected nodes are resolved to their full markdown content
3. **Answer Generation**: Retrieved context is sent to the answer model

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

### Query Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tree-search-model` | claude-haiku | Model for tree navigation |
| `--answer-model` | llama-4-maverick | Model for answer generation |
| `--verbose` | false | Show reasoning and context |

## URL Filtering

The AI URL filter uses a customizable prompt to determine which URLs are relevant for crawling. The prompt is located at `prompts/url_filtering_prompt.txt`.

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

- **URL Filtering**: Claude Haiku 4.5
- **Summary Generation**: Claude Haiku 4.5 or Llama 4 Maverick
- **Tree Search**: Claude Haiku 4.5
- **Answer Generation**: Llama 4 Maverick

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
