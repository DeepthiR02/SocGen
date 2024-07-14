import re
from oletools.olevba import VBA_Parser
import networkx as nx
import matplotlib.pyplot as plt
from textwrap import wrap
from ml_model import classify_code, predict_complexity, summarize_code, translate_to_python, generate_flowchart_text
from concurrent.futures import ThreadPoolExecutor
import functools

# Use a thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=5)


def memoize(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


@memoize
def extract_vba(file_path):
    vba_parser = VBA_Parser(file_path)
    vba_modules = []

    if vba_parser.detect_vba_macros():
        vba_modules = list(vba_parser.extract_macros())
    else:
        return "No VBA macros found in the file"

    vba_parser.close()
    return vba_modules


def analyze_code_structure(vba_code):
    functions = len(re.findall(r'\b(Function|Sub)\s+\w+', vba_code))
    variables = len(set(re.findall(r'\b(Dim|Public|Private)\s+(\w+)', vba_code)))
    if_count = vba_code.count("If ")
    for_count = vba_code.count("For ")
    while_count = vba_code.count("While ")
    do_count = vba_code.count("Do ")
    select_count = vba_code.count("Select Case")

    return {
        "functions": functions,
        "variables": variables,
        "if_statements": if_count,
        "for_loops": for_count,
        "while_loops": while_count,
        "do_loops": do_count,
        "select_case": select_count
    }


def process_module(vba_filename, vba_code):
    documentation = f"Module: {vba_filename}\n"
    documentation += f"{'=' * len(vba_filename)}\n\n"

    code_structure = analyze_code_structure(vba_code)
    documentation += f"Code Structure:\n"
    for key, value in code_structure.items():
        documentation += f"- {key.replace('_', ' ').title()}: {value}\n"
    documentation += "\n"

    # Run ML tasks concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_summary = executor.submit(summarize_code, vba_code)
        future_classify = executor.submit(classify_code, vba_code)
        future_complexity = executor.submit(predict_complexity, vba_code)
        future_translate = executor.submit(translate_to_python, vba_code)

        documentation += f"Code Summary:\n{future_summary.result()}\n\n"
        documentation += f"Code Classification:\n{future_classify.result()}\n\n"
        documentation += f"Code Complexity Score:\n{future_complexity.result():.2f}\n\n"
        documentation += f"Equivalent Python Code:\n{future_translate.result()}\n\n"

    documentation += "-" * 50 + "\n\n"
    return documentation


@memoize
def generate_flowchart(code):
    flowchart_text = generate_flowchart_text(code)

    G = nx.DiGraph()
    lines = flowchart_text.split('\n')
    node_id = 0
    stack = []
    node_colors = {}
    node_sizes = {}

    for line in lines:
        line = line.strip()
        if line:
            G.add_node(node_id, label=line)
            node_colors[node_id] = '#AED6F1'  # Light Blue
            node_sizes[node_id] = 2500
            if stack:
                G.add_edge(stack[-1], node_id)
            stack.append(node_id)
            node_id += 1

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.9, iterations=50)

    nx.draw_networkx_nodes(G, pos, node_color=[node_colors[node] for node in G.nodes()],
                           node_size=[node_sizes[node] for node in G.nodes()], ax=ax)

    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, ax=ax)

    labels = nx.get_node_attributes(G, 'label')
    wrapped_labels = {node: '\n'.join(wrap(label, 20)) for node, label in labels.items()}
    nx.draw_networkx_labels(G, pos, wrapped_labels, font_size=6, ax=ax)

    ax.set_axis_off()
    plt.tight_layout()

    flowchart_path = 'static/flowchart.png'
    fig.savefig(flowchart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return flowchart_path


def generate_documentation(file_path):
    try:
        vba_modules = extract_vba(file_path)

        if isinstance(vba_modules, str):
            return vba_modules, None

        # Process modules concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_module, vba_filename, vba_code)
                       for _, _, vba_filename, vba_code in vba_modules]
            results = [future.result() for future in futures]

        documentation = "VBA Macro Documentation\n\n" + "".join(results)

        # Generate flowchart for the first module
        flowchart_path = generate_flowchart(vba_modules[0][3])

        return documentation, flowchart_path
    except Exception as e:
        return str(e), None