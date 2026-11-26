"""Regression checks using synthetic fixtures only; no original data required."""
import json
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'breast-cancer-classification.ipynb'
CELLS = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']

def source(index):
    return ''.join(CELLS[index]['source'])

def run(namespace, *indices):
    for i in indices:
        exec(compile(source(i), f'{NOTEBOOK.name}:cell-{i}', 'exec'), namespace)

class NotebookTests(unittest.TestCase):
    def test_python_cells_compile(self):
        for i, cell in enumerate(CELLS):
            if cell['cell_type'] != 'code':
                continue
            # Notebook magics are handled by IPython, not the Python parser.
            code = '\n'.join(line for line in source(i).splitlines() if not line.lstrip().startswith(('%', '!')))
            with self.subTest(cell=i):
                compile(code, f'{NOTEBOOK.name}:cell-{i}', 'exec')

    def test_oversampling_preserves_holdout(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import RandomOverSampler
        from collections import Counter
        data = pd.DataFrame({'Age': range(30, 50), 'Type': ['a', 'b'] * 10, 'Status': ['Alive'] * 15 + ['Dead'] * 5})
        ns = dict(data=data, StandardScaler=StandardScaler, LabelEncoder=LabelEncoder, train_test_split=train_test_split, RandomOverSampler=RandomOverSampler)
        run(ns, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23)
        self.assertEqual(len(ns['x_test']), 4)
        self.assertEqual(len(ns['y_test']), 4)
        self.assertEqual(len(set(Counter(ns['y_res']).values())), 1)
        self.assertGreater(len(ns['y_res']), len(ns['y_train']))

if __name__ == '__main__':
    unittest.main()
