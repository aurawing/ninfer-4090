"""CPU-only exact C++/Python recipe-stream parity through the built harness."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'quality'))
import kvmem_quality as q
from owned_process import run_owned

with tempfile.TemporaryDirectory(prefix='ninfer-quality-render-') as folder:
    base = Path(folder)
    cases = [case for lines in (80, 87, 259) for task in q.TASKS for case in q.make_cases(task, 4096, 1, lines)]
    q.write_new(base / 'request.json', {'operation': 'render', 'cases': cases})
    run_owned([sys.argv[1], str(base / 'request.json'), str(base / 'rendered.jsonl')],
              stdout=base / 'stdout.txt', stderr=base / 'stderr.txt', timeout=30)
    rendered = [json.loads(line)['case'] for line in (base / 'rendered.jsonl').read_text(encoding='utf-8').splitlines()]
    if rendered != cases:
        raise AssertionError('C++/Python recipe stream differs byte-for-byte')
    for case in rendered:
        q.audit_case(case)
    print('36 exact C++/Python recipe parity cases passed; no Engine/model load')
