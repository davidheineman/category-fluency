import torch
import matplotlib.pyplot as plt


def calculate_entropy(probs, e=1e-10):
    if not isinstance(probs, torch.Tensor):
        probs = torch.Tensor(probs)

    probs = torch.clamp(probs, min=e, max=1-e)
    return -torch.sum(probs * torch.log2(probs))


def render_table(results):
    """
    Converts a nested dict to a LaTeX table.

    Input:
        {
            "Column 1": {
                "Row 1": ...,
                "Row 2": ...,
                ...
            },
            "Column 2": {
                "Row 1": ...,
                "Row 2": ...,
                ...
            },
            ...
        }
    
    Output:
        | Column 1 | Column 2 | ...      |
        |----------|----------|----------|
        | Row 1    | Row 1    | ...      |
        | Row 2    | Row 2    | ...      |
        | ...      | ...      | ...      |

    """
    if not (isinstance(results, dict) and all(isinstance(value, dict) and set(value.keys()) == set(next(iter(results.values())).keys()) for value in results.values())):
        raise ValueError("Input must be a nested dict with matching inner keys")

    headers = list(results[list(results.keys())[0]].keys())

    header_spec = ''.join(['C{1.5cm}' for _ in headers])
    header_text = ' & '.join([f'\\textbf{{{key}}}' for key in headers])

    result_text = '\n'
    for entry_key, entry_item in results.items():
        entry_text = f'{entry_key} & '
        entry_text += ' & '.join([str(round(entry_item[h], 2)) if isinstance(entry_item[h], float) else str(entry_item[h]) for h in headers])
        entry_text += ' \\\\\n'
        result_text += entry_text

    table = f"""\\begin{{tabular}}{{p{{3cm}}{header_spec}}}
\\toprule
\\textbf{{Method}} & {header_text} \\\\
\\midrule
{result_text}
\\bottomrule
\\end{{tabular}}
"""

    return table


def visualize_exemplar_dist(exemplars, probs, prev_exemplars=['animal'], colors=None):
    plt.figure(figsize=(7, 2))
    plt.plot(exemplars, probs, color='black')
    plt.fill_between(exemplars, probs, color='red', alpha=0.3)
    plt.ylim(0, max(probs) + 0.01)
    plt.xlim(0, len(exemplars) - 1)
    plt.xticks(rotation=45, fontsize=7, ha='right')
    plt.xlabel(r'$x_i$')
    plt.ylabel(r'$p(x_i|x_{<i})$')
    plt.title(r'$p(x_i|' + "\,\,".join(prev_exemplars) + ')$')

    if colors:
        for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
            label.set_color(colors[i])

    plt.show()