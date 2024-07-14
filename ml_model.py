from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, RobertaTokenizer, \
    T5ForConditionalGeneration
import torch
from functools import lru_cache
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Use CodeBERT for classification and complexity
clf_model_name = "microsoft/codebert-base"
tokenizer_clf = AutoTokenizer.from_pretrained(clf_model_name)
model_clf = AutoModelForSequenceClassification.from_pretrained(clf_model_name, num_labels=5)

# Use CodeT5 for summarization and translation
codet5_model_name = "Salesforce/codet5-small"
tokenizer_codet5 = RobertaTokenizer.from_pretrained(codet5_model_name)
model_codet5 = T5ForConditionalGeneration.from_pretrained(codet5_model_name)

# Use BART for code explanation
explanation_model = pipeline("text2text-generation", model="facebook/bart-large-cnn")


@lru_cache(maxsize=128)
def classify_code(code: str) -> str:
    inputs = tokenizer_clf(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model_clf(**inputs).logits
    predicted_class_id = logits.argmax().item()
    classes = ["Other", "Function", "Class", "Module", "Script"]
    return classes[predicted_class_id]


@lru_cache(maxsize=128)
def predict_complexity(code: str) -> float:
    inputs = tokenizer_clf(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        complexity_score = model_clf(**inputs).logits.mean().item()
    return complexity_score


@lru_cache(maxsize=128)
def summarize_code(code: str, max_length: int = 150, min_length: int = 30) -> str:
    def preprocess(text):
        # Remove comments
        text = re.sub("'.*", "", text)
        text = re.sub("REM .*", "", text)
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Simple stopword removal
        stop_words = set(['and', 'or', 'the', 'a', 'an', 'in', 'is', 'it', 'to', 'for', 'of', 'as', 'by', 'with'])
        return " ".join([token for token in tokens if token not in stop_words])

    def extract_components(text):
        return {
            'subs': len(re.findall(r'Sub\s+(\w+)', text)),
            'functions': len(re.findall(r'Function\s+(\w+)', text)),
            'variables': len(re.findall(r'Dim\s+(\w+)', text)),
            'loops': len(re.findall(r'\b(For|Do While|Do Until)\b', text)),
            'conditionals': len(re.findall(r'\bIf\b', text))
        }

    # Preprocess the code
    preprocessed_code = preprocess(code)

    # Extract key components
    components = extract_components(code)

    # Use TF-IDF to identify important terms
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_code])
    feature_names = vectorizer.get_feature_names_out()
    importance_scores = np.squeeze(tfidf_matrix.toarray())
    top_indices = importance_scores.argsort()[-5:][::-1]
    important_terms = [feature_names[i] for i in top_indices]

    # Identify the main purpose (simple heuristic)
    purposes = ['update', 'calculate', 'format', 'print', 'display']
    purpose = next((p for p in purposes if p in preprocessed_code), 'perform various operations')

    # Generate summary
    summary_parts = [
        f"This code contains {components['subs']} subroutines and {components['functions']} functions.",
        f"It uses {components['variables']} variables, {components['loops']} loops, and {components['conditionals']} conditionals.",
        f"Key terms: {', '.join(important_terms)}.",
        f"Main purpose appears to be to {purpose}."
    ]

    # Combine parts and adjust length
    summary = " ".join(summary_parts)
    if len(summary) < min_length:
        summary += " The code may require further analysis for a more detailed understanding."
    elif len(summary) > max_length:
        summary = summary[:max_length].rsplit(' ', 1)[0] + '...'

    return summary


@lru_cache(maxsize=128)
def explain_code_functionality(code: str) -> str:
    prompt = f"Explain this code in plain English:\n\n{code}\n\nExplanation:"
    explanation = explanation_model(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
    return explanation.split("Explanation:")[1].strip()


@lru_cache(maxsize=128)
def translate_to_python(vba_code: str) -> str:
    # Dictionary of common VBA to Python translations
    translations = {
        # Function and Sub definitions
        r'(Public\s+)?(Function|Sub)\s+(\w+)\s*\((.*?)\)(?:\s+As\s+(\w+))?': lambda
            m: f"def {m.group(3)}({translate_parameters(m.group(4))}):{' # -> ' + m.group(5) if m.group(5) else ''}",
        r'End (Function|Sub)': '',

        # Variable declarations
        r'Dim\s+(\w+)(?:\s+As\s+(\w+))?': lambda
            m: f"{m.group(1)} = None  # Type in VBA: {m.group(2) if m.group(2) else 'Variant'}",
        r'(Public|Private)\s+(\w+)(?:\s+As\s+(\w+))?': lambda
            m: f"{m.group(2)} = None  # {m.group(1)} variable, Type in VBA: {m.group(3) if m.group(3) else 'Variant'}",

        # Updated control structures
        r'If\s+(.*?)\s+Then': r'if \1:',
        r'ElseIf\s+(.*?)\s+Then': r'elif \1:',
        r'Else\s*If\s+(.*?)\s+Then': r'elif \1:',  # New pattern for "Else If"
        r'Else': r'else:',
        r'End If': '',
        r'For\s+(\w+)\s+=\s+(\S+)\s+To\s+(\S+)(?:\s+Step\s+(\S+))?': lambda
            m: f"for {m.group(1)} in range({m.group(2)}, {m.group(3)}{',' + m.group(4) if m.group(4) else ''}):",
        r'Next': '',
        r'Do\s+While\s+(.*?)': r'while \1:',
        r'Do\s+Until\s+(.*?)': r'while not (\1):',
        r'Loop': '',
        r'Exit (For|Do|Function|Sub)': r'break  # Exit \1 in VBA',

        # Error handling
        r'On Error Resume Next': 'import contextlib\nwith contextlib.suppress(Exception):',
        r'On Error GoTo 0': '# Error handling disabled',
        r'On Error GoTo (\w+)': '# Error handling: GoTo not directly translatable to Python',

        # Common functions and methods
        r'MsgBox\s*\((.*?)\)': r'print(\1)  # MsgBox in VBA',
        r'Debug\.Print\s+(.+)': r'print(\1)',
        r'(\w+)\.Select': r'# \1.Select() - No direct equivalent in Python',
        r'Set\s+(\w+)\s+=\s+(\S+)': r'\1 = \2',

        # Operators
        r'\<\>': '!=',
        r'&': '+',  # For string concatenation

        # Comments
        r'\'': r'#',
        r'REM\s': r'# ',
    }

    def translate_parameters(params: str) -> str:
        """Translate VBA parameter declarations to Python."""
        if not params.strip():
            return ""
        param_list = params.split(',')
        python_params = []
        for param in param_list:
            parts = param.strip().split()
            if len(parts) >= 3 and parts[-2].lower() == 'as':
                python_params.append(f"{parts[0]}: '{parts[-1]}'")
            else:
                python_params.append(parts[0])
        return ', '.join(python_params)

    python_code = vba_code
    for vba_pattern, python_pattern in translations.items():
        if callable(python_pattern):
            python_code = re.sub(vba_pattern, python_pattern, python_code, flags=re.IGNORECASE)
        else:
            python_code = re.sub(vba_pattern, python_pattern, python_code, flags=re.IGNORECASE)

    # Add import statements for common VBA functions
    python_code = "import math\nimport random\n\n" + python_code

    # Replace VBA's built-in constants
    python_code = python_code.replace("vbNewLine", "\\n")
    python_code = python_code.replace("vbTab", "\\t")
    python_code = python_code.replace("vbNullString", "''")

    return python_code


@lru_cache(maxsize=128)
def generate_flowchart_text(code: str) -> str:
    def extract_structure(code):
        lines = code.split('\n')
        structure = []
        indent_level = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('Function '):
                structure.append(('function', stripped, indent_level))
            elif stripped.startswith('If '):
                structure.append(('if', stripped, indent_level))
                indent_level += 1
            elif stripped.startswith('ElseIf ') or stripped.startswith('Else If '):
                indent_level -= 1
                structure.append(('elif', stripped, indent_level))
                indent_level += 1
            elif stripped == 'Else':
                indent_level -= 1
                structure.append(('else', stripped, indent_level))
                indent_level += 1
            elif stripped == 'End If':
                indent_level -= 1
            elif stripped.startswith('For ') or stripped.startswith('Do '):
                structure.append(('loop', stripped, indent_level))
                indent_level += 1
            elif stripped == 'Next' or stripped == 'Loop':
                indent_level -= 1
            elif '=' in stripped and not stripped.startswith('If '):
                structure.append(('assignment', stripped, indent_level))
        return structure

    def generate_flowchart_description(structure):
        description = ["```mermaid", "graph TD"]
        node_counter = 0
        node_stack = ['A']
        description.append("A[Start]")

        for item_type, content, level in structure:
            node_counter += 1
            current_node = f"N{node_counter}"

            if item_type == 'function':
                func_name = content.split()[1].split('(')[0]
                description.append(f"{node_stack[-1]} --> {current_node}[Function: {func_name}]")
                node_stack.append(current_node)
            elif item_type == 'if':
                condition = content.split('If ')[1].split(' Then')[0]
                description.append(f"{node_stack[-1]} --> {current_node}{{Condition: {condition}}}")
                node_counter += 1
                yes_node = f"N{node_counter}"
                description.append(f"{current_node} -->|Yes| {yes_node}[Execute if block]")
                node_stack.append(yes_node)
            elif item_type == 'elif':
                condition = content.split('If ')[1].split(' Then')[0]
                description.append(f"{node_stack.pop()} --> {current_node}{{Condition: {condition}}}")
                node_counter += 1
                yes_node = f"N{node_counter}"
                description.append(f"{current_node} -->|Yes| {yes_node}[Execute elif block]")
                node_stack.append(yes_node)
            elif item_type == 'else':
                description.append(f"{node_stack.pop()} --> {current_node}[Execute else block]")
                node_stack.append(current_node)
            elif item_type == 'loop':
                loop_type = content.split()[0]
                description.append(f"{node_stack[-1]} --> {current_node}[{loop_type} loop]")
                node_counter += 1
                condition_node = f"N{node_counter}"
                description.append(f"{current_node} --> {condition_node}{{Loop condition}}")
                node_counter += 1
                body_node = f"N{node_counter}"
                description.append(f"{condition_node} -->|True| {body_node}[Execute loop body]")
                description.append(f"{body_node} --> {condition_node}")
                node_stack.append(condition_node)
            elif item_type == 'assignment':
                description.append(f"{node_stack[-1]} --> {current_node}[{content}]")
                node_stack.append(current_node)

        description.append(f"{node_stack[-1]} --> Z[End]")
        description.append("```")
        return "\n".join(description)

    structure = extract_structure(code)
    flowchart_text = generate_flowchart_description(structure)

    return flowchart_text


def generate_user_friendly_summary(code: str) -> str:
    structure = generate_flowchart_text(code).split('\n')[2:-1]  # Remove mermaid syntax
    summary = ["This code does the following:"]

    for line in structure:
        if "Function:" in line:
            summary.append(f"- Defines a function called {line.split('Function: ')[1][:-1]}")
        elif "Condition:" in line:
            summary.append(f"- Checks a condition: {line.split('Condition: ')[1][:-1]}")
        elif "Execute if block" in line:
            summary.append("  - If the condition is true, it does something")
        elif "Execute elif block" in line:
            summary.append("  - If the previous condition was false, it checks another condition")
        elif "Execute else block" in line:
            summary.append("  - If none of the conditions were true, it does something else")
        elif "loop" in line.lower():
            summary.append(f"- Repeats a set of actions ({line.split('[')[1][:-1]})")
        elif "=" in line and "[" in line and "]" in line:
            summary.append(f"- Assigns a value: {line.split('[')[1][:-1]}")

    return "\n".join(summary)
