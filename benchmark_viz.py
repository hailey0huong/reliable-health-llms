"""Convert synthetic benchmark JSON to a styled HTML visualization."""
import json
import argparse
from pathlib import Path
import html


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synthetic Benchmark Viewer</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --warning-color: #d97706;
            --danger-color: #dc2626;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, #059669, #10b981);
            color: white;
            border-radius: 12px;
        }}
        
        header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }}
        
        .stat-item {{
            background: rgba(255,255,255,0.2);
            padding: 0.5rem 1rem;
            border-radius: 8px;
        }}
        
        .navigation {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex-wrap: wrap;
        }}
        
        .nav-btn {{
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            background: var(--primary-color);
            color: white;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .nav-btn:hover:not(:disabled) {{
            background: #1d4ed8;
            transform: translateY(-1px);
        }}
        
        .nav-btn:disabled {{
            background: #cbd5e1;
            cursor: not-allowed;
        }}
        
        .nav-btn .key-hint {{
            font-size: 0.75rem;
            opacity: 0.8;
            background: rgba(255,255,255,0.2);
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
        }}
        
        .question-select {{
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: 2px solid var(--border-color);
            font-size: 1rem;
            min-width: 350px;
            cursor: pointer;
            background: white;
        }}
        
        .question-select:focus {{
            outline: none;
            border-color: var(--primary-color);
        }}
        
        .question-counter {{
            font-weight: 600;
            color: var(--text-muted);
            min-width: 100px;
            text-align: center;
        }}
        
        .filter-section {{
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }}
        
        .filter-btn {{
            padding: 0.5rem 1rem;
            border-radius: 20px;
            border: 2px solid var(--border-color);
            background: white;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s;
        }}
        
        .filter-btn:hover {{
            border-color: var(--primary-color);
        }}
        
        .filter-btn.active {{
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }}
        
        .filter-btn.answerable.active {{
            background: var(--success-color);
            border-color: var(--success-color);
        }}
        
        .filter-btn.hard_but_fair.active {{
            background: var(--warning-color);
            border-color: var(--warning-color);
        }}
        
        .filter-btn.boundary_tests.active {{
            background: var(--danger-color);
            border-color: var(--danger-color);
        }}
        
        .benchmark-card {{
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            overflow: hidden;
            border: 1px solid var(--border-color);
            display: none;
        }}
        
        .benchmark-card.active {{
            display: block;
        }}
        
        .card-header {{
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        
        .card-header.answerable {{
            background: linear-gradient(to right, #dcfce7, #d1fae5);
            border-left: 4px solid var(--success-color);
        }}
        
        .card-header.hard_but_fair {{
            background: linear-gradient(to right, #fef3c7, #fde68a);
            border-left: 4px solid var(--warning-color);
        }}
        
        .card-header.boundary_tests {{
            background: linear-gradient(to right, #fee2e2, #fecaca);
            border-left: 4px solid var(--danger-color);
        }}
        
        .card-number {{
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--text-color);
        }}
        
        .bucket-badge {{
            display: inline-block;
            padding: 0.375rem 0.875rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }}
        
        .bucket-badge.answerable {{
            background: var(--success-color);
            color: white;
        }}
        
        .bucket-badge.hard_but_fair {{
            background: var(--warning-color);
            color: white;
        }}
        
        .bucket-badge.boundary_tests {{
            background: var(--danger-color);
            color: white;
        }}
        
        .card-body {{
            padding: 1.5rem;
        }}
        
        .prompt-section {{
            margin-bottom: 1.5rem;
        }}
        
        .section-label {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        
        .prompt-text {{
            background: #f8fafc;
            padding: 1.25rem;
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
            font-size: 1rem;
            white-space: pre-wrap;
            line-height: 1.7;
        }}
        
        .correct-answer {{
            background: linear-gradient(135deg, #dcfce7, #d1fae5);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #86efac;
            margin-bottom: 1.5rem;
        }}
        
        .correct-answer-label {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--success-color);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}
        
        .correct-answer-text {{
            font-weight: 700;
            color: #166534;
            font-size: 1.1rem;
        }}
        
        .collapsible-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            background: #f1f5f9;
            border-radius: 8px;
            cursor: pointer;
            user-select: none;
            margin-bottom: 0.5rem;
            transition: background 0.2s;
        }}
        
        .collapsible-header:hover {{
            background: #e2e8f0;
        }}
        
        .collapsible-title {{
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--text-muted);
        }}
        
        .collapsible-icon {{
            font-size: 0.75rem;
            transition: transform 0.2s;
        }}
        
        .collapsible-header.collapsed .collapsible-icon {{
            transform: rotate(-90deg);
        }}
        
        .collapsible-content {{
            padding: 1rem;
            background: #fafafa;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            transition: max-height 0.3s ease-out, opacity 0.3s ease-out;
        }}
        
        .collapsible-content.collapsed {{
            max-height: 0 !important;
            padding: 0;
            border: none;
            opacity: 0;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        
        .metadata-item {{
            background: white;
            padding: 0.75rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }}
        
        .metadata-label {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}
        
        .metadata-value {{
            font-weight: 500;
            font-size: 0.9rem;
            word-break: break-word;
        }}
        
        .question-context {{
            background: white;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            font-size: 0.9rem;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }}
        
        .extracted-info {{
            margin-top: 1rem;
        }}
        
        .condition-item {{
            background: white;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
        }}
        
        .condition-text {{
            flex: 1;
            font-size: 0.9rem;
        }}
        
        .condition-badges {{
            display: flex;
            gap: 0.5rem;
            flex-shrink: 0;
        }}
        
        .weight-badge {{
            background: linear-gradient(135deg, var(--primary-color), #7c3aed);
            color: white;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }}
        
        .category-badge {{
            background: #e0e7ff;
            color: #3730a3;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
        }}
        
        .scroll-top-btn {{
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: var(--primary-color);
            color: white;
            border: none;
            cursor: pointer;
            font-size: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }}
        
        .scroll-top-btn:hover {{
            transform: scale(1.1);
        }}
        
        .llm-answer {{
            background: linear-gradient(135deg, #dbeafe, #e0e7ff);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #93c5fd;
            margin-top: 1rem;
        }}
        
        .llm-answer-label {{
            font-size: 0.75rem;
            font-weight: 600;
            color: #1e40af;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        
        .llm-answer-text {{
            font-weight: 600;
            color: #1e3a8a;
            margin-bottom: 0.5rem;
        }}
        
        .llm-justification {{
            font-size: 0.9rem;
            color: #3730a3;
        }}
        
        .model-response-section {{
            margin-bottom: 1.5rem;
        }}
        
        .model-response-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}
        
        .model-response-text {{
            background: #f1f5f9;
            padding: 1.25rem;
            border-radius: 8px;
            border-left: 4px solid #6366f1;
            font-size: 1rem;
            white-space: pre-wrap;
            line-height: 1.7;
        }}
        
        .model-info-bar {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
            padding: 0.75rem 1rem;
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            flex-wrap: wrap;
        }}
        
        .model-name-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.375rem 0.75rem;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        
        .correctness-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
            padding: 0.375rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 700;
        }}
        
        .correctness-badge.correct {{
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
            color: #166534;
            border: 1px solid #86efac;
        }}
        
        .correctness-badge.incorrect {{
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            color: #991b1b;
            border: 1px solid #fca5a5;
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}
            
            .navigation {{
                flex-direction: column;
            }}
            
            .question-select {{
                min-width: 100%;
            }}
            
            .nav-btn .key-hint {{
                display: none;
            }}
            
            .metadata-grid {{
                grid-template-columns: 1fr;
            }}
            
            .card-header {{
                flex-direction: column;
                align-items: flex-start;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧪 Benchmark Viewer</h1>
            <p>USMLE Clinical Question Patient Prompts</p>
            <div class="stats">
                <div class="stat-item">
                    <strong>{total_items}</strong> Items
                </div>
                <div class="stat-item">
                    <strong>{answerable_count}</strong> Answerable
                </div>
                <div class="stat-item">
                    <strong>{hard_count}</strong> Hard but Fair
                </div>
                <div class="stat-item">
                    <strong>{boundary_count}</strong> Boundary Tests
                </div>
            </div>
        </header>
        
        <div class="filter-section">
            <button class="filter-btn active" onclick="filterByBucket('all')">All</button>
            <button class="filter-btn answerable" onclick="filterByBucket('answerable')">Answerable</button>
            <button class="filter-btn hard_but_fair" onclick="filterByBucket('hard_but_fair')">Hard but Fair</button>
            <button class="filter-btn boundary_tests" onclick="filterByBucket('boundary_tests')">Boundary Tests</button>
        </div>
        
        <nav class="navigation">
            <button class="nav-btn" id="prevBtn" onclick="showPrevItem()">
                ← Prev <span class="key-hint">←</span>
            </button>
            
            <select class="question-select" id="itemSelect" onchange="showItem(this.value)">
                {options_html}
            </select>
            
            <span class="question-counter" id="itemCounter">1 / {total_items}</span>
            
            <button class="nav-btn" id="nextBtn" onclick="showNextItem()">
                Next → <span class="key-hint">→</span>
            </button>
        </nav>
        
        <main id="itemsContainer">
            {items_html}
        </main>
    </div>
    
    <button class="scroll-top-btn" onclick="window.scrollTo({{top: 0, behavior: 'smooth'}})" title="Go to top">↑</button>
    
    <script>
        let currentIndex = 0;
        let currentFilter = 'all';
        let visibleIndices = [];
        const totalItems = {total_items};
        
        function updateVisibleIndices() {{
            visibleIndices = [];
            document.querySelectorAll('.benchmark-card').forEach((card, idx) => {{
                const bucket = card.dataset.bucket;
                if (currentFilter === 'all' || bucket === currentFilter) {{
                    visibleIndices.push(idx);
                }}
            }});
        }}
        
        function filterByBucket(bucket) {{
            currentFilter = bucket;
            
            // Update filter buttons
            document.querySelectorAll('.filter-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            updateVisibleIndices();
            
            // Update dropdown options visibility
            const select = document.getElementById('itemSelect');
            Array.from(select.options).forEach((opt, idx) => {{
                const card = document.getElementById('item-' + idx);
                if (card) {{
                    const cardBucket = card.dataset.bucket;
                    opt.style.display = (bucket === 'all' || cardBucket === bucket) ? '' : 'none';
                }}
            }});
            
            // Show first visible item
            if (visibleIndices.length > 0) {{
                showItem(visibleIndices[0]);
            }}
        }}
        
        function showItem(index) {{
            index = parseInt(index);
            currentIndex = index;
            
            // Hide all items
            document.querySelectorAll('.benchmark-card').forEach(card => {{
                card.classList.remove('active');
            }});
            
            // Show selected item
            const activeCard = document.getElementById('item-' + index);
            if (activeCard) {{
                activeCard.classList.add('active');
            }}
            
            // Update dropdown
            document.getElementById('itemSelect').value = index;
            
            // Update counter
            const visiblePos = visibleIndices.indexOf(index) + 1;
            document.getElementById('itemCounter').textContent = visiblePos + ' / ' + visibleIndices.length;
            
            // Update button states
            const currentVisibleIndex = visibleIndices.indexOf(index);
            document.getElementById('prevBtn').disabled = currentVisibleIndex <= 0;
            document.getElementById('nextBtn').disabled = currentVisibleIndex >= visibleIndices.length - 1;
            
            // Scroll to top
            window.scrollTo({{top: 0, behavior: 'smooth'}});
        }}
        
        function showNextItem() {{
            const currentVisibleIndex = visibleIndices.indexOf(currentIndex);
            if (currentVisibleIndex < visibleIndices.length - 1) {{
                showItem(visibleIndices[currentVisibleIndex + 1]);
            }}
        }}
        
        function showPrevItem() {{
            const currentVisibleIndex = visibleIndices.indexOf(currentIndex);
            if (currentVisibleIndex > 0) {{
                showItem(visibleIndices[currentVisibleIndex - 1]);
            }}
        }}
        
        function toggleCollapsible(element) {{
            element.classList.toggle('collapsed');
            const content = element.nextElementSibling;
            content.classList.toggle('collapsed');
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            if (e.key === 'ArrowLeft') {{
                e.preventDefault();
                showPrevItem();
            }} else if (e.key === 'ArrowRight') {{
                e.preventDefault();
                showNextItem();
            }}
        }});
        
        // Initialize
        updateVisibleIndices();
        showItem(0);
    </script>
</body>
</html>
"""


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if text is None:
        return ""
    return html.escape(str(text))


def format_bucket_name(bucket: str) -> str:
    """Format bucket name for display."""
    return bucket.replace("_", " ").title()


def generate_extracted_info_html(extracted_info: list) -> str:
    """Generate HTML for extracted clinical information."""
    if not extracted_info:
        return "<p style='color: var(--text-muted); font-style: italic;'>No extracted information available.</p>"
    
    items_html = ""
    for info in extracted_info:
        condition = escape_html(info.get('condition', info.get('condition_text', 'N/A')))
        weight = info.get('weight', 'N/A')
        bucket = info.get('bucket', 'N/A')
        
        items_html += f'''
        <div class="condition-item">
            <div class="condition-text">{condition}</div>
            <div class="condition-badges">
                <span class="weight-badge">W: {weight}</span>
                <span class="category-badge">{bucket}</span>
            </div>
        </div>
        '''
    
    return f'<div class="extracted-info">{items_html}</div>'


def generate_item_html(item: dict, index: int) -> str:
    """Generate HTML for a single benchmark item."""
    prompt = escape_html(item.get('prompt', ''))
    bucket = item.get('bucket', 'unknown')
    metadata = item.get('metadata', {})
    
    # Extract metadata fields
    question_no = metadata.get('question_no', index + 1)
    question_type = metadata.get('type_of_question', 'N/A')
    step = metadata.get('step', 'N/A')
    correct_response = escape_html(metadata.get('correct_response', 'N/A'))
    question_context = escape_html(metadata.get('question_context', ''))
    options = escape_html(metadata.get('options', ''))
    
    # Nested metadata
    nested_meta = metadata.get('metadata', {})
    topic = nested_meta.get('topic', 'N/A')
    medical_code = nested_meta.get('medical_code', 'N/A')
    medical_code_name = nested_meta.get('medical_code_name', 'N/A')
    medical_code_desc = escape_html(nested_meta.get('medical_code_description', 'N/A'))
    
    # Extracted info and LLM answer
    extracted_info = metadata.get('llm_extracted_info', [])
    llm_answer = metadata.get('llm_final_answer', {})
    
    extracted_html = generate_extracted_info_html(extracted_info)
    
    # LLM answer HTML
    llm_answer_html = ""
    if llm_answer:
        answer = escape_html(llm_answer.get('answer', 'N/A'))
        justification = escape_html(llm_answer.get('justification', ''))
        llm_answer_html = f'''
        <div class="llm-answer">
            <div class="llm-answer-label">🤖 LLM Answer</div>
            <div class="llm-answer-text">{answer}</div>
            <div class="llm-justification">{justification}</div>
        </div>
        '''
    
    # Build model info bar with response, model_name, and is_correct
    model_info_html = ""
    has_model_info = 'model_name' in item or 'is_correct' in item
    if has_model_info:
        model_info_parts = []
        if 'model_name' in item:
            model_info_parts.append(f'<span class="model-name-badge">🤖 {escape_html(item["model_name"])}</span>')
        if 'is_correct' in item:
            is_correct = item['is_correct']
            if is_correct:
                model_info_parts.append('<span class="correctness-badge correct">✓ Correct</span>')
            else:
                model_info_parts.append('<span class="correctness-badge incorrect">✗ Incorrect</span>')
        model_info_html = f'<div class="model-info-bar">{"".join(model_info_parts)}</div>'
    
    # Model response section
    response_html = ""
    if 'response' in item:
        response_html = f'''
        <div class="model-response-section">
            <div class="section-label">🤖 Model Response</div>
            <div class="model-response-text">{escape_html(item["response"])}</div>
        </div>
        '''
    
    bucket_css = bucket.replace("-", "_")
    
    return f'''
    <article class="benchmark-card" id="item-{index}" data-bucket="{bucket}">
        <div class="card-header {bucket_css}">
            <span class="card-number">#{question_no} • Step {step} • {question_type}</span>
            <span class="bucket-badge {bucket_css}">{format_bucket_name(bucket)}</span>
        </div>
        <div class="card-body">
            {model_info_html}
            
            <div class="prompt-section">
                <div class="section-label">💬 Patient Prompt</div>
                <div class="prompt-text">{prompt}</div>
            </div>
            
            {response_html}
            
            <div class="correct-answer">
                <div class="correct-answer-label">✓ Correct Answer</div>
                <div class="correct-answer-text">{correct_response}</div>
            </div>
            
            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                <span class="collapsible-title">📋 Original Question & Metadata</span>
                <span class="collapsible-icon">▼</span>
            </div>
            <div class="collapsible-content collapsed">
                <div class="metadata-grid">
                    <div class="metadata-item">
                        <div class="metadata-label">Topic</div>
                        <div class="metadata-value">{topic}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Medical Code</div>
                        <div class="metadata-value">{medical_code}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Code Name</div>
                        <div class="metadata-value">{medical_code_name}</div>
                    </div>
                </div>
                <div class="metadata-item" style="margin-bottom: 1rem;">
                    <div class="metadata-label">Code Description</div>
                    <div class="metadata-value" style="font-size: 0.85rem;">{medical_code_desc}</div>
                </div>
                <div class="section-label" style="margin-top: 1rem;">Original Question Context</div>
                <div class="question-context">{question_context}</div>
                <div class="section-label" style="margin-top: 1rem;">Answer Options</div>
                <div class="question-context">{options}</div>
            </div>
            
            <div class="collapsible-header collapsed" onclick="toggleCollapsible(this)">
                <span class="collapsible-title">🔬 Extracted Clinical Information ({len(extracted_info)} items)</span>
                <span class="collapsible-icon">▼</span>
            </div>
            <div class="collapsible-content collapsed">
                {extracted_html}
                {llm_answer_html}
            </div>
        </div>
    </article>
    '''


def convert_benchmark_to_html(input_file: str, output_file: str = None):
    """Convert a synthetic benchmark JSON to an HTML visualization."""
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    # Count buckets
    bucket_counts = {"answerable": 0, "hard_but_fair": 0, "boundary_tests": 0}
    for item in data:
        bucket = item.get('bucket', 'unknown')
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1
    
    # Generate HTML for each item
    items_html = ""
    dropdown_options_html = ""
    
    for idx, item in enumerate(data):
        items_html += generate_item_html(item, idx)
        
        # Build dropdown option
        bucket = item.get('bucket', 'unknown')
        question_no = item.get('metadata', {}).get('question_no', idx + 1)
        prompt_preview = item.get('prompt', '')[:60]
        bucket_icon = {"answerable": "✅", "hard_but_fair": "⚠️", "boundary_tests": "🔴"}.get(bucket, "❓")
        dropdown_options_html += f'<option value="{idx}">{bucket_icon} #{question_no}: {escape_html(prompt_preview)}...</option>\n'
    
    # Fill in the template
    html_content = HTML_TEMPLATE.format(
        total_items=len(data),
        answerable_count=bucket_counts["answerable"],
        hard_count=bucket_counts["hard_but_fair"],
        boundary_count=bucket_counts["boundary_tests"],
        items_html=items_html,
        options_html=dropdown_options_html
    )
    
    # Determine output file path
    if output_file is None:
        output_file = input_path.with_suffix('.html')
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Successfully converted {len(data)} benchmark items to HTML")
    print(f"📄 Output saved to: {output_file}")
    print(f"   - Answerable: {bucket_counts['answerable']}")
    print(f"   - Hard but Fair: {bucket_counts['hard_but_fair']}")
    print(f"   - Boundary Tests: {bucket_counts['boundary_tests']}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert synthetic benchmark JSON to a styled HTML visualization"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output HTML file (defaults to input filename with .html extension)",
        default=None
    )
    
    args = parser.parse_args()
    convert_benchmark_to_html(args.input_file, args.output)


# python benchmark_viz.py data/cardiology_usmle_synthetic_benchmark_v1.json -o data/benchmark_viz.html
# python benchmark_viz.py results/benchmark_raw_results_gpt-5.2.json -o results/benchmark_viz_gpt-5.2.html
# python benchmark_viz.py results/raw_results/benchmark_raw_results_gpt-oss-120b.json -o visualization/benchmark_viz_gpt-oss-120b.html
if __name__ == "__main__":
    main()
