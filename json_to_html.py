"""Convert USMLE questions JSON to a styled HTML visualization."""
import json
import argparse
from pathlib import Path


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>USMLE Questions Viewer</title>
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
            background: linear-gradient(135deg, var(--primary-color), #7c3aed);
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
            min-width: 300px;
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
        
        .question-card {{
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            overflow: hidden;
            border: 1px solid var(--border-color);
            display: none;
        }}
        
        .question-card.active {{
            display: block;
        }}
        
        .question-header {{
            background: linear-gradient(to right, #f1f5f9, #e2e8f0);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        
        .question-number {{
            font-weight: 700;
            font-size: 1.25rem;
            color: var(--primary-color);
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge-step {{
            background: #dbeafe;
            color: #1e40af;
        }}
        
        .badge-type {{
            background: #f3e8ff;
            color: #7c3aed;
        }}
        
        .badge-topic {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .question-body {{
            padding: 1.5rem;
        }}
        
        .question-text {{
            font-size: 1rem;
            margin-bottom: 1.5rem;
            white-space: pre-wrap;
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
        }}
        
        .options-list {{
            list-style: none;
            margin-bottom: 1.5rem;
        }}
        
        .options-list li {{
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 8px;
            background: #f8fafc;
            border: 1px solid var(--border-color);
            transition: all 0.2s;
        }}
        
        .options-list li.correct {{
            background: #dcfce7;
            border-color: var(--success-color);
            font-weight: 600;
        }}
        
        .options-list li.correct::before {{
            content: "✓ ";
            color: var(--success-color);
        }}
        
        .section-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-muted);
            margin: 1.5rem 0 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .section-title::before {{
            content: "";
            display: inline-block;
            width: 4px;
            height: 1rem;
            background: var(--primary-color);
            border-radius: 2px;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
        }}
        
        .metadata-item {{
            padding: 0.5rem;
        }}
        
        .metadata-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .metadata-value {{
            font-weight: 500;
            margin-top: 0.25rem;
        }}
        
        .extracted-info {{
            display: grid;
            gap: 0.75rem;
        }}
        
        .condition-card {{
            background: #f8fafc;
            border-radius: 8px;
            padding: 1rem;
            border-left: 4px solid var(--primary-color);
        }}
        
        .condition-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }}
        
        .condition-text {{
            font-weight: 500;
            flex: 1;
        }}
        
        .condition-badges {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        
        .weight-badge {{
            background: linear-gradient(135deg, var(--primary-color), #7c3aed);
            color: white;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .bucket-badge {{
            background: #fef3c7;
            color: #92400e;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
        }}
        
        .condition-explanation {{
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }}
        
        .final-answer {{
            background: linear-gradient(135deg, #dcfce7, #d1fae5);
            border-radius: 8px;
            padding: 1.25rem;
            border: 1px solid #86efac;
        }}
        
        .final-answer-text {{
            font-size: 1.125rem;
            font-weight: 700;
            color: var(--success-color);
            margin-bottom: 0.75rem;
        }}
        
        .justification {{
            font-size: 0.9rem;
            color: #166534;
        }}
        
        .collapsible {{
            cursor: pointer;
            user-select: none;
        }}
        
        .collapsible:hover {{
            opacity: 0.8;
        }}
        
        .collapsible::after {{
            content: " ▼";
            font-size: 0.75rem;
        }}
        
        .collapsible.collapsed::after {{
            content: " ▶";
        }}
        
        .collapsible-content {{
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        
        .collapsible-content.collapsed {{
            max-height: 0 !important;
        }}
        
        .patient-prompts-container {{
            display: grid;
            gap: 1.5rem;
        }}
        
        .prompt-category {{
            background: #f8fafc;
            border-radius: 8px;
            padding: 1rem;
            border-left: 4px solid var(--primary-color);
        }}
        
        .prompt-category.answerable {{
            border-left-color: var(--success-color);
        }}
        
        .prompt-category.hard-but-fair {{
            border-left-color: var(--warning-color);
        }}
        
        .prompt-category.boundary-tests {{
            border-left-color: var(--danger-color);
        }}
        
        .prompt-category-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }}
        
        .prompt-category-title {{
            font-weight: 600;
            font-size: 0.95rem;
        }}
        
        .prompt-category-badge {{
            padding: 0.125rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.7rem;
            font-weight: 600;
        }}
        
        .prompt-category.answerable .prompt-category-badge {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .prompt-category.hard-but-fair .prompt-category-badge {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .prompt-category.boundary-tests .prompt-category-badge {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        .prompt-list {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        
        .prompt-item {{
            background: white;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            font-size: 0.9rem;
            position: relative;
            padding-left: 2rem;
        }}
        
        .prompt-item::before {{
            content: "💬";
            position: absolute;
            left: 0.5rem;
            top: 0.75rem;
        }}
        
        .prompt-item.empty {{
            color: var(--text-muted);
            font-style: italic;
        }}
        
        .prompt-item.empty::before {{
            content: "";
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
        
        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}
            
            .question-header {{
                flex-direction: column;
                align-items: flex-start;
            }}
            
            .metadata-grid {{
                grid-template-columns: 1fr;
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
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📋 USMLE Questions Viewer</h1>
            <p>Clinical Question Analysis & Extraction Results</p>
            <div class="stats">
                <div class="stat-item">
                    <strong>{total_questions}</strong> Questions
                </div>
                <div class="stat-item">
                    <strong>{topics}</strong> Topics
                </div>
            </div>
        </header>
        
        <nav class="navigation">
            <button class="nav-btn" id="prevBtn" onclick="showPrevQuestion()">
                ← Prev <span class="key-hint">←</span>
            </button>
            
            <select class="question-select" id="questionSelect" onchange="showQuestion(this.value)">
                {options_html}
            </select>
            
            <span class="question-counter" id="questionCounter">1 / {total_questions}</span>
            
            <button class="nav-btn" id="nextBtn" onclick="showNextQuestion()">
                Next → <span class="key-hint">→</span>
            </button>
        </nav>
        
        <main id="questionsContainer">
            {questions_html}
        </main>
    </div>
    
    <button class="scroll-top-btn" onclick="window.scrollTo({{top: 0, behavior: 'smooth'}})" title="Go to top">↑</button>
    
    <script>
        let currentIndex = 0;
        const totalQuestions = {total_questions};
        
        function showQuestion(index) {{
            index = parseInt(index);
            currentIndex = index;
            
            // Hide all questions
            document.querySelectorAll('.question-card').forEach(card => {{
                card.classList.remove('active');
            }});
            
            // Show selected question
            const activeCard = document.getElementById('question-' + index);
            if (activeCard) {{
                activeCard.classList.add('active');
            }}
            
            // Update dropdown
            document.getElementById('questionSelect').value = index;
            
            // Update counter
            document.getElementById('questionCounter').textContent = (index + 1) + ' / ' + totalQuestions;
            
            // Update button states
            document.getElementById('prevBtn').disabled = index === 0;
            document.getElementById('nextBtn').disabled = index === totalQuestions - 1;
            
            // Scroll to top
            window.scrollTo({{top: 0, behavior: 'smooth'}});
        }}
        
        function showNextQuestion() {{
            if (currentIndex < totalQuestions - 1) {{
                showQuestion(currentIndex + 1);
            }}
        }}
        
        function showPrevQuestion() {{
            if (currentIndex > 0) {{
                showQuestion(currentIndex - 1);
            }}
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            // Don't trigger if user is typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            if (e.key === 'ArrowLeft') {{
                e.preventDefault();
                showPrevQuestion();
            }} else if (e.key === 'ArrowRight') {{
                e.preventDefault();
                showNextQuestion();
            }}
        }});
        
        // Collapsible sections
        document.querySelectorAll('.collapsible').forEach(el => {{
            el.addEventListener('click', function() {{
                this.classList.toggle('collapsed');
                const content = this.nextElementSibling;
                content.classList.toggle('collapsed');
            }});
        }});
        
        // Initialize first question
        showQuestion(0);
    </script>
</body>
</html>
"""


def parse_options(options_text: str) -> list:
    """Parse options string into a list of individual options."""
    if not options_text:
        return []
    # Split by newline and filter empty lines
    options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
    return options


def get_option_letter(option: str) -> str:
    """Extract the letter from an option like '(A) Some text'."""
    if option.startswith('(') and len(option) > 2:
        return option[1]
    return ''


def generate_patient_prompts_html(patient_prompts: dict) -> str:
    """Generate HTML for patient prompts section."""
    if not patient_prompts:
        return "<p style='color: var(--text-muted);'>No patient prompts available.</p>"
    
    categories = [
        ("answerable", "Answerable", "✅ Contains sufficient information to answer correctly"),
        ("hard_but_fair", "Hard But Fair", "⚠️ Challenging but still answerable with careful reasoning"),
        ("boundary_tests", "Boundary Tests", "🔴 Edge cases to test model limitations"),
    ]
    
    html = '<div class="patient-prompts-container">'
    
    for key, title, description in categories:
        prompts = patient_prompts.get(key, [])
        css_class = key.replace("_", "-")
        
        html += f'''
        <div class="prompt-category {css_class}">
            <div class="prompt-category-header">
                <span class="prompt-category-title">{title}</span>
                <span class="prompt-category-badge">{len(prompts)} prompts</span>
            </div>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem;">{description}</p>
            <div class="prompt-list">
        '''
        
        if prompts:
            for prompt in prompts:
                if prompt:  # Skip empty strings
                    html += f'<div class="prompt-item">{prompt}</div>'
        else:
            html += '<div class="prompt-item empty">No prompts in this category</div>'
        
        html += '</div></div>'
    
    html += '</div>'
    return html


def generate_question_html(item: dict, index: int) -> str:
    """Generate HTML for a single question."""
    question_no = item.get('question_no', index + 1)
    question_type = item.get('type_of_question', 'Unknown')
    step = item.get('step', 'N/A')
    question_text = item.get('question_context', item.get('question', ''))
    options_text = item.get('options', '')
    correct_response = item.get('correct_response', '')
    metadata = item.get('metadata', {})
    extracted_info = item.get('llm_extracted_info', [])
    final_answer = item.get('llm_final_answer', {})
    patient_prompts = item.get('patient_prompts', {})
    
    # Parse options
    options = parse_options(options_text)
    correct_letter = get_option_letter(correct_response) if correct_response else ''
    
    # Build options HTML
    options_html = ""
    for opt in options:
        opt_letter = get_option_letter(opt)
        is_correct = opt_letter == correct_letter
        css_class = "correct" if is_correct else ""
        options_html += f'<li class="{css_class}">{opt}</li>\n'
    
    # Build metadata HTML
    topic = metadata.get('topic', 'N/A')
    medical_code = metadata.get('medical_code', 'N/A')
    medical_code_name = metadata.get('medical_code_name', 'N/A')
    medical_code_desc = metadata.get('medical_code_description', 'N/A')
    
    metadata_html = f"""
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
    <div class="metadata-item" style="margin-top: 0.5rem;">
        <div class="metadata-label">Code Description</div>
        <div class="metadata-value" style="font-size: 0.875rem;">{medical_code_desc}</div>
    </div>
    """
    
    # Build extracted info HTML
    extracted_html = ""
    if extracted_info:
        for info in extracted_info:
            condition = info.get('condition', info.get('condition_text', 'N/A'))
            weight = info.get('weight', 'N/A')
            bucket = info.get('bucket', 'N/A')
            explanation = info.get('explanation', '')
            rationale = info.get('rationale', '')
            confidence = info.get('confidence', '')
            
            extracted_html += f"""
            <div class="condition-card">
                <div class="condition-header">
                    <div class="condition-text">{condition}</div>
                    <div class="condition-badges">
                        <span class="weight-badge">Weight: {weight}</span>
                        <span class="bucket-badge">{bucket}</span>
                    </div>
                </div>
                <div class="condition-explanation">
                    {f'<strong>Explanation for weight:</strong> {explanation}' if explanation else ''}
                    {f'<br><strong>Rationale for label:</strong> {rationale}' if rationale else ''}
                    {f'<br><strong>Confidence for label:</strong> {confidence}' if confidence else ''}
                </div>
            </div>
            """
    else:
        extracted_html = "<p style='color: var(--text-muted);'>No extracted information available.</p>"
    
    # Build final answer HTML
    final_answer_html = ""
    if final_answer:
        answer = final_answer.get('answer', 'N/A')
        justification = final_answer.get('justification', '')
        final_answer_html = f"""
        <div class="final-answer">
            <div class="final-answer-text">🎯 {answer}</div>
            <div class="justification">{justification}</div>
        </div>
        """
    else:
        final_answer_html = "<p style='color: var(--text-muted);'>No final answer available.</p>"
    
    # Build patient prompts HTML
    patient_prompts_html = generate_patient_prompts_html(patient_prompts)
    
    # Count total prompts
    total_prompts = sum(len(patient_prompts.get(k, [])) for k in ['answerable', 'hard_but_fair', 'boundary_tests'])
    
    # Use index for the id to match JavaScript
    return f"""
    <article class="question-card" id="question-{index}">
        <div class="question-header">
            <span class="question-number">Question {question_no}</span>
            <div>
                <span class="badge badge-step">Step {step}</span>
                <span class="badge badge-type">{question_type}</span>
                <span class="badge badge-topic">{topic}</span>
            </div>
        </div>
        <div class="question-body">
            <div class="question-text">{question_text}</div>
            
            <h3 class="section-title">Answer Options</h3>
            <ul class="options-list">
                {options_html}
            </ul>
            
            <h3 class="section-title collapsible">Metadata</h3>
            <div class="collapsible-content">
                {metadata_html}
            </div>
            
            <h3 class="section-title collapsible">Extracted Clinical Information ({len(extracted_info)} items)</h3>
            <div class="collapsible-content">
                <div class="extracted-info">
                    {extracted_html}
                </div>
            </div>
            
            <h3 class="section-title">LLM Final Answer</h3>
            {final_answer_html}
            
            <h3 class="section-title collapsible">Patient Prompts ({total_prompts} total)</h3>
            <div class="collapsible-content">
                {patient_prompts_html}
            </div>
        </div>
    </article>
    """


def convert_json_to_html(input_file: str, output_file: str = None):
    """Convert a JSON file of USMLE questions to an HTML visualization."""
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    # Generate HTML for each question
    questions_html = ""
    dropdown_options_html = ""
    topics = set()
    
    for idx, item in enumerate(data):
        questions_html += generate_question_html(item, idx)
        topic = item.get('metadata', {}).get('topic', 'Unknown')
        topics.add(topic)
        
        # Build dropdown option
        question_no = item.get('question_no', idx + 1)
        question_preview = item.get('question_context', item.get('question', ''))[:80]
        dropdown_options_html += f'<option value="{idx}">Q{question_no}: {question_preview}...</option>\n'
    
    # Fill in the template
    html_content = HTML_TEMPLATE.format(
        total_questions=len(data),
        topics=len(topics),
        questions_html=questions_html,
        options_html=dropdown_options_html
    )
    
    # Determine output file path
    if output_file is None:
        output_file = input_path.with_suffix('.html')
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Successfully converted {len(data)} questions to HTML")
    print(f"Output saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert USMLE questions JSON to a styled HTML visualization"
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
    convert_json_to_html(args.input_file, args.output)

# python -m json_to_html data/cardiology_usmle_rewritten_v1.json -o data/cardiology_usmle_rewritten_v1.html
if __name__ == "__main__":
    main()
