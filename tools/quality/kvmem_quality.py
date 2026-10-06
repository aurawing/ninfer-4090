"""Frozen, local RULER-style synthetic checks; stdlib only, not corpus acceptance.

Draft generation and tokenizer fitting precede freeze. No answer-dependent fixture
selection is supported. See README.md for the batch harness and run lifecycle.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys

VERSION = 'kvmem-local-quality-v1'
SEED = 20261006
TASKS = ('needle', 'repeat', 'multi', 'chain', 'frequency', 'summary')
ACK = 'Document received. I will use it when you ask your question.'
PREFIX = 'LOCAL SYNTHETIC DOCUMENT\n'
SUFFIX = '\nEND OF DOCUMENT'
POLICIES = {'enable_thinking': False, 'preserve_thinking': True,
            'temperature': 0, 'top_k': 0, 'top_p': 1, 'min_p': 0,
            'presence_penalty': 0, 'frequency_penalty': 0, 'sampling_seed': SEED,
            'allow_prefix_reuse': True, 'short_output_budget': 256,
            'summary_output_budget': 1024, 'summary_include_model_defaults': False,
            'short_include_model_defaults': True, 'mtp_draft_tokens': 3,
            'proposal_head': 'optimized', 'concurrency': 1,
            'non_dense_mtp_window_tokens': 32768, 'vision': False,
            'prompt_guard_tokens': 128, 'rubric': VERSION,
            'summary_rubric': 'exact anchored entity | field | value; all ten required'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_case_id(case_id):
    # One portable filename component; exclude separators, drives, dot traversal,
    # Windows device names, and characters with filesystem-specific semantics.
    if (not isinstance(case_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', case_id)
            or case_id.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                                   *(f'LPT{i}' for i in range(1, 10))}):
        raise ValueError('case ID must be a safe single filename component')
    return case_id


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, indent=2, ensure_ascii=False)
        out.write('\n')


def background_units(task, index):
    rng = random.Random(SEED + 700000 + TASKS.index(task) * 100000 + index)
    subjects = ('walkers', 'gardeners', 'visitors', 'painters', 'readers', 'neighbors', 'volunteers', 'craftspeople',
                'students', 'birdwatchers', 'curators', 'travelers', 'musicians', 'librarians', 'bakers', 'potters')
    actions = ('observed', 'described', 'discussed', 'remembered', 'sketched', 'arranged', 'examined', 'photographed')
    objects = ('the courtyard', 'the footpath', 'the meadow', 'the hillside', 'the riverside', 'the orchard',
               'the gallery', 'the garden', 'the terrace', 'the reading room', 'the workshop', 'the market')
    qualities = ('quiet', 'bright', 'shaded', 'welcoming', 'peaceful', 'spacious', 'familiar', 'colorful',
                 'sheltered', 'pleasant', 'rustic', 'lively')
    endings = ('while clouds drifted overhead', 'as a breeze moved through the trees',
               'before returning to their usual activities', 'during a leisurely afternoon',
               'with a shared interest in the surroundings', 'after listening to distant birds')
    templates = ('The {s} {a} {o}, which appeared {q}, {e}.\n',
                 'In casual conversation the {s} {a} {o} and called it {q}, {e}.\n',
                 'A {q} atmosphere surrounded {o}; the {s} {a} the scene {e}.\n',
                 'Near {o}, the {s} {a} the {q} surroundings {e}.\n')
    return [rng.choice(templates).format(s=rng.choice(subjects), a=rng.choice(actions), o=rng.choice(objects),
                                        q=rng.choice(qualities), e=rng.choice(endings)) for _ in range(128)]


def render_document(recipe):
    count = recipe['filler_lines']
    inserts = recipe['injections']
    # Stable separated injection fractions; neutral filler is the only fit variable.
    out = [PREFIX]
    previous = 0
    for index, injection in enumerate(inserts):
        position = ((index + 1) * count) // (len(inserts) + 1)
        units = recipe['filler_units']
        out.append(''.join(units[line % len(units)] for line in range(previous, position)))
        out.append(injection + '\n')
        previous = position
    out.extend([''.join(units[line % len(units)] for line in range(previous, count)), SUFFIX])
    return ''.join(out)


def document_of(case):
    content = case['messages'][0]['content']
    return content.split('\n\nQUESTION\n', 1)[0] if case['form'] == 'single' else content


def set_document(case, document):
    case['messages'][0]['content'] = (document + '\n\nQUESTION\n' + case['question']
                                      if case['form'] == 'single' else document)


def stamp_case(case):
    case['document_sha256'] = sha_bytes(document_of(case).encode('utf-8'))
    case['input_sha256'] = sha_bytes(canonical({'messages': case['messages'],
                                              'current_input_message': case['current_input_message'],
                                              'policies': POLICIES}))
    case['truth_sha256'] = sha_bytes(canonical(case['truth']))
    recipe = {key: value for key, value in case['recipe'].items() if key != 'filler_lines'}
    case['blueprint_sha256'] = sha_bytes(canonical({'seed': case['seed'], 'recipe': recipe,
                                                  'truth': case['truth'], 'question': case['question'], 'policies': POLICIES}))


def task_data(task, index, query_index=None):
    rng = random.Random(SEED + TASKS.index(task) * 100000 + index)
    refs = [str(v) for v in rng.sample(range(10000000, 99999999), 12)]
    suffix = ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(6))
    keys = [f'archive-{suffix}-{j}{tail}' for j in range(4) for tail in ('O', '0')]
    if task in ('needle', 'multi'):
        entries = list(zip(keys, refs))
        injections = [f'REFERENCE {key} = {value}' for key, value in entries]
        if task == 'needle':
            target = keys[2 * ((index if query_index is None else query_index) % 4)]
            answer = {'reference': dict(entries)[target]}
            question = f'What is the eight-digit reference for the exact key {target}? Return only JSON {{"reference":"eight-digit value"}}.'
        else:
            target = keys[::2]
            answer = {'references': {key: dict(entries)[key] for key in target}}
            question = 'Retrieve the exact references for these four keys: ' + ', '.join(target) + '. Return only JSON {"references":{"exact-key":"eight-digit value",...}}.'
        return injections, question, {'entries': entries, 'target': target, 'answer': answer}
    if task == 'repeat':
        target = keys[0]
        entries = [(target, refs[j]) for j in range(4)]
        injections = []
        for j, (key, value) in enumerate(entries):
            injections.extend([f'REFERENCE {key} = {value}', f'REFERENCE {keys[2*j+1]} = {refs[4+j]}'])
        question = f'Retrieve ALL four distinct eight-digit references assigned to the exact key {target}, in document order. Return only JSON {{"references":["value",...]}}.'
        return injections, question, {'entries': entries, 'target': target,
                                     'distractors': list(zip(keys[1::2], refs[4:8])),
                                     'answer': {'references': refs[:4]}}
    if task == 'chain':
        roots = [rng.randrange(10000, 20000), rng.randrange(20000, 30000), rng.randrange(30000, 40000)]
        assignments = [(f'{prefix}{j}', str(number) if j == 1 else f'{prefix}{j-1}')
                       for prefix, number in zip(('X', 'Y', 'Z'), roots) for j in range(1, 6)]
        rng.shuffle(assignments)
        question = f'Follow all assignments. List ALL variables whose resolved numeric value equals {roots[0]}. Return only JSON {{"variables":["name",...]}} in numeric variable order.'
        return [f'ASSIGN {key}={value}' for key, value in assignments], question, {
            'assignments': assignments, 'target_number': roots[0],
            'answer': {'variables': [f'X{j}' for j in range(1, 6)]}}
    if task == 'frequency':
        markers = [f'marker{suffix.lower()}{tail}' for tail in ('amber', 'birch', 'cobalt', 'dahlia', 'elm', 'flint')]
        counts = rng.sample(range(11, 48), 6)  # No tied ranks, frozen before inference.
        words = [word for word, count in zip(markers, counts) for _ in range(count)]
        rng.shuffle(words)
        injections = ['MARKERS ' + ' '.join(words[j:j+7]) for j in range(0, len(words), 7)]
        top = sorted(markers, key=lambda word: -counts[markers.index(word)])[:3]
        question = 'Among only these marker words, return the TOP THREE by total document frequency, most frequent first: ' + ', '.join(markers) + '. Return only JSON {"top3":["word",...]}.'
        return injections, question, {'markers': markers, 'counts': dict(zip(markers, counts)), 'answer': {'top3': top}}
    if task == 'summary':
        fields = ('opening year', 'lead engineer', 'storage capacity', 'survey district', 'launch month',
                  'primary material', 'inspection cycle', 'funding source', 'supply depot', 'tracking code')
        values = (str(rng.randrange(2040, 2090)), f'Mira {suffix}', f'{rng.randrange(101, 999)} crates',
                  f'North {suffix}', 'November', 'ceramic', 'seventeen days',
                  f'Foundation {suffix}', f'Depot {suffix}', f'TR{refs[0]}')
        facts = [[f'Project {suffix}-{j+1:02}', field, value] for j, (field, value) in enumerate(zip(fields, values))]
        question = ('Write a detailed 1024-token summary of the document, covering all ten planted project facts. '
                    'For each fact include one exact line Entity | field | value, copied accurately from the document. '
                    'Add explanatory prose to develop the summary. Do not combine values from different projects.')
        return ['FACT ' + ' | '.join(fact) for fact in facts], question, {'facts': facts}
    raise ValueError('unknown task')


def make_cases(task, context, count=5, filler_lines=1000):
    cases = []
    for index in range(count):
        injections, question, truth = task_data(task, index)
        truth = json.loads(canonical(truth))
        recipe = {'filler_lines': filler_lines, 'injections': injections,
                  'filler_units': background_units(task, index), 'prefix': PREFIX, 'suffix': SUFFIX,
                  'placement': 'even-fractions-v1'}
        document = render_document(recipe)
        for form in ('single', 'historical'):
            case = {'id': f'{task}-c{context}-n{index:02}-{form}', 'task': task,
                    'context_cap': context, 'case_index': index, 'form': form,
                    'document_id': f'{task}-c{context}-n{index:02}', 'seed': SEED,
                    'recipe': recipe, 'question': question, 'truth': truth,
                    'messages': ([{'role': 'user', 'content': document + '\n\nQUESTION\n' + question}]
                                 if form == 'single' else [{'role': 'user', 'content': document},
                                      {'role': 'assistant', 'content': ACK}, {'role': 'user', 'content': question}]),
                    'current_input_message': 0 if form == 'single' else 2,
                    'requested_output_tokens': 1024 if task == 'summary' else 256,
                    'include_model_defaults': task != 'summary'}
            stamp_case(case)
            cases.append(case)
    return cases


def audit_case(case):
    """Parse actual document independently; hashes alone cannot validate truth."""
    validate_case_id(case.get('id'))
    if case.get('form') not in ('single', 'historical'):
        raise ValueError('unknown input form; expected single or historical')
    for field in ('case_index', 'context_cap'):
        value = case.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if field == 'case_index' else 1):
            raise ValueError('invalid semantic case metadata: ' + field)
    task, truth, doc = case['task'], case['truth'], document_of(case)
    if task not in TASKS or case['seed'] != SEED:
        raise ValueError('unrecognized task or seed')
    expected_injections, expected_question, expected_truth = task_data(task, case['case_index'], case.get('query_index'))
    if canonical(expected_truth) != canonical(truth) or case['question'] != expected_question:
        raise ValueError('truth/question differs from deterministic seed recipe')
    if case['recipe']['injections'] != expected_injections or doc != render_document(case['recipe']):
        raise ValueError('document differs from frozen neutral filler/insertion recipe')
    filler_units = background_units(task, case['case_index'])
    filler_text = ''.join(filler_units)
    if case['recipe']['filler_units'] != filler_units or case['recipe']['prefix'] != PREFIX or case['recipe']['suffix'] != SUFFIX:
        raise ValueError('filler/template identity changed')
    wanted_messages = ([{'role': 'user', 'content': doc + '\n\nQUESTION\n' + case['question']}]
                       if case['form'] == 'single' else [{'role': 'user', 'content': doc},
                            {'role': 'assistant', 'content': ACK}, {'role': 'user', 'content': case['question']}])
    if case['messages'] != wanted_messages or case['current_input_message'] != (0 if case['form'] == 'single' else 2):
        raise ValueError('input form/current query boundary changed')
    if case['requested_output_tokens'] != (1024 if task == 'summary' else 256) or case['include_model_defaults'] != (task != 'summary'):
        raise ValueError('output/stop policy changed')
    stamped = dict(case)
    stamp_case(stamped)
    if any(stamped[key] != case[key] for key in ('document_sha256', 'input_sha256', 'truth_sha256', 'blueprint_sha256')):
        raise ValueError('case SHA mismatch')
    # Every non-neutral line must belong to exactly one recognized record grammar.
    neutral_lines = {unit.strip() for unit in filler_units} | {PREFIX.strip(), SUFFIX.strip()}
    lines = [line for line in doc.splitlines() if line and line not in neutral_lines]
    if task in ('needle', 'multi', 'repeat'):
        entries = []
        for line in lines:
            match = re.fullmatch(r'REFERENCE ([\w-]+) = ([0-9]{8})', line)
            if not match:
                raise ValueError('unrecognized reference or leaked answer in document')
            entries.append(match.groups())
        expected = truth['entries'] + truth.get('distractors', [])
        if Counter(entries) != Counter(tuple(e) for e in expected):
            raise ValueError('reference injection frequencies differ')
        if task in ('needle', 'multi') and (len(entries) != 8 or len(set(k for k, _ in entries)) != 8):
            raise ValueError('expected eight unique keys including four lookalikes')
        if task == 'repeat' and [v for k, v in entries if k == truth['target']] != truth['answer']['references']:
            raise ValueError('four separated repeated references differ')
        for _, value in entries:
            if value in case['question'] or value in filler_text:
                raise ValueError('reference answer leaked outside records')
    elif task == 'chain':
        graph = {}
        for line in lines:
            match = re.fullmatch(r'ASSIGN ([XYZ][1-5])=([XYZ][1-5]|[0-9]+)', line)
            if not match or match[1] in graph:
                raise ValueError('invalid/duplicate graph node')
            graph[match[1]] = match[2]
        if graph != dict(truth['assignments']):
            raise ValueError('reference graph differs')
        def resolve(key, seen):
            if key in seen or key not in graph:
                raise ValueError('cycle or missing graph node')
            value = graph[key]
            return int(value) if value.isdigit() else resolve(value, seen | {key})
        actual = sorted((key for key in graph if resolve(key, set()) == truth['target_number']), key=lambda key: (key[0], int(key[1:])))
        if actual != truth['answer']['variables']:
            raise ValueError('independently resolved variable set differs')
    elif task == 'frequency':
        if any(not line.startswith('MARKERS ') for line in lines):
            raise ValueError('invalid marker record')
        counters = Counter(word for line in lines for word in line[8:].split())
        if dict(counters) != truth['counts'] or len(set(counters.values())) != len(counters):
            raise ValueError('independent marker counters differ or rank ties')
        if [word for word, _ in counters.most_common(3)] != truth['answer']['top3']:
            raise ValueError('marker rank differs')
    else:
        facts = [line[5:].split(' | ') for line in lines if line.startswith('FACT ')]
        if len(facts) != 10 or facts != truth['facts'] or len(lines) != 10:
            raise ValueError('ten planted facts differ')
        if any(value in case['question'] or value in filler_text for _, _, value in facts):
            raise ValueError('fact value leaked outside document')
    return True


def strict_json(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non-JSON constant')))


def validate(case, text):
    result = {'validator_version': VERSION, 'quality_pass': False}
    if case['task'] != 'summary':
        try:
            answer = strict_json(text)
            result['quality_pass'] = answer == case['truth']['answer']
            result['reason'] = 'exact JSON match' if result['quality_pass'] else 'JSON answer differs from complete truth'
        except (ValueError, TypeError) as error:
            result['reason'] = 'invalid strict JSON: ' + str(error)
        return result
    facts = case['truth']['facts']
    expected = {(entity, field): value for entity, field, value in facts}
    seen = defaultdict(list)
    unknown = []
    for line in text.splitlines():
        # Whole anchored line; a loose value substring never counts as a fact.
        match = re.fullmatch(r'\s*(?:[-*] )?(Project [A-Z]+-[0-9]{2})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*', line)
        if match:
            entity, field, value = match.groups()
            if (entity, field) in expected:
                seen[(entity, field)].append(value)
            else:
                unknown.append([entity, field, value])
    covered = [[entity, field, value] for entity, field, value in facts
               if seen[(entity, field)] == [value]]
    wrong = [[entity, field, values] for (entity, field), values in seen.items()
             if values != [expected[(entity, field)]]]
    result.update(fact_coverage_count=len(covered), fact_coverage_fraction=len(covered)/10,
                  all_ten_facts_complete=len(covered) == 10 and not wrong and not unknown,
                  covered_facts=covered, wrong_or_duplicate_links=wrong, unknown_links=unknown)
    result['quality_pass'] = result['all_ten_facts_complete']
    result['reason'] = 'all ten exact anchored facts' if result['quality_pass'] else 'missing/wrong/duplicate anchored fact links'
    return result


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = ['tools/quality/kvmem_quality.py', 'tools/quality/owned_process.py',
             'bench/kvmem_quality_bench.cpp', 'tests/test_kvmem_quality.py',
             'tests/test_kvmem_quality_render.py', 'tools/quality/README.md']
    return {name: sha_file(root / name) for name in paths if (root / name).is_file()}


def freeze(cases, folder, identity):
    folder = Path(folder).resolve()
    if folder.exists():
        raise FileExistsError('freeze destination must be new (no overwrite)')
    for field in ('tokenizer_artifact_sha256', 'counter_binary_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', identity.get(field, '')):
            raise ValueError('freeze requires exact artifact/counter SHA identity')
    ids = [validate_case_id(case.get('id')) for case in cases]
    if len(ids) != len({case_id.casefold() for case_id in ids}):
        raise ValueError('duplicate case identity')
    files = []
    intended_paths = []
    # Preflight EVERY path before creating the freeze root or writing any case.
    for case_id in ids:
        relative = 'cases/' + case_id + '.json'
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or path.parent != folder / 'cases':
            raise ValueError('intended frozen case path escapes freeze root')
        intended_paths.append((relative, path))
    for case in cases:
        audit_case(case)
        count = case.get('prompt_tokens')
        if not isinstance(count, int) or count <= 0 or count + case['requested_output_tokens'] + POLICIES['prompt_guard_tokens'] > case['context_cap']:
            raise ValueError('actual tokenizer count absent or insufficient output headroom')
    folder.mkdir(parents=True)
    for case, (relative, path) in zip(cases, intended_paths):
        write_new(path, case)
        files.append({'path': relative, 'id': case['id'], 'sha256': sha_file(path)})
    manifest = {'version': VERSION, 'seed': SEED, 'policies': POLICIES,
                'identity': identity, 'source_sha256': source_identity(), 'case_count': len(cases),
                'cases': files, 'synthetic_only': True}
    write_new(folder / 'manifest.json', manifest)
    write_new(folder / 'manifest-sha256.json', {'sha256': sha_file(folder / 'manifest.json')})
    return manifest


def load_frozen(folder):
    folder = Path(folder)
    manifest_path = folder / 'manifest.json'
    marker = json.loads((folder / 'manifest-sha256.json').read_text(encoding='utf-8'))
    if sha_file(manifest_path) != marker['sha256']:
        raise ValueError('frozen manifest SHA differs')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['version'] != VERSION or manifest['policies'] != POLICIES or manifest['source_sha256'] != source_identity():
        raise ValueError('frozen rubric/policy/source identity differs')
    cases = []
    for entry in manifest['cases']:
        path = folder / entry['path']
        if not path.resolve().is_relative_to(folder.resolve()) or sha_file(path) != entry['sha256']:
            raise ValueError('frozen case path/SHA differs')
        case = json.loads(path.read_text(encoding='utf-8'))
        if case['id'] != entry['id']:
            raise ValueError('frozen case ID differs')
        audit_case(case)
        cases.append(case)
    if len(cases) != manifest['case_count']:
        raise ValueError('frozen case count differs')
    return manifest, cases


def profiles():
    rows = []
    for priority, cap, view, reference in ((1, 262144, 32768, 'tiered-int8'),
                                          (2, 262144, 131072, 'dense-rk4'),
                                          (2, 131072, 131072, 'dense-rk4'),
                                          (3, 131072, 32768, 'tiered-int8')):
        for denominator in ('kept-band', 'all-committed'):
            rows.append({'id': f'c{cap}-kvmem-int8-v{view}-{denominator}', 'priority': priority,
                         'context_cap': cap, 'mode': 'kvmem', 'dtype': 'int8', 'requested_view_tokens': view,
                         'denominator': denominator, 'reference_id': f'c{cap}-{reference}'})
        ref = {'id': f'c{cap}-{reference}', 'priority': priority, 'context_cap': cap,
               'mode': 'tiered-exact' if reference == 'tiered-int8' else 'dense',
               'dtype': 'int8' if reference == 'tiered-int8' else 'rk4v4-e8',
               'requested_view_tokens': 32768 if reference == 'tiered-int8' else cap,
               'denominator': 'not-applicable', 'reference_id': None}
        if ref['id'] not in [row['id'] for row in rows]:
            rows.append(ref)
    for view in (32768, 131072):
        for denominator in ('kept-band', 'all-committed'):
            rows.append({'id': f'c262144-kvmem-rk4-v{view}-{denominator}', 'priority': 4,
                         'context_cap': 262144, 'mode': 'kvmem', 'dtype': 'rk4v4-e8',
                         'requested_view_tokens': view, 'denominator': denominator,
                         'reference_id': f'c262144-kvmem-int8-v{view}-{denominator}',
                         'comparison_policy': 'same-requested-token-counts'})
    return rows


def validate_normal_suite_pairs(cases):
    """Planning is the normal-suite boundary; isolated pilot freezes remain valid."""
    ids = [validate_case_id(case.get('id')) for case in cases]
    if len(ids) != len({case_id.casefold() for case_id in ids}):
        raise ValueError('duplicate normal-suite case ID')
    groups = defaultdict(list)
    for case in cases:
        audit_case(case)
        semantic = (case['task'], case['context_cap'], case['case_index'])
        groups[semantic].append(case)
    document_owners = {}
    common_fields = ('document_id', 'seed', 'question', 'truth_sha256', 'blueprint_sha256',
                     'document_sha256', 'requested_output_tokens', 'include_model_defaults', 'query_index')
    for semantic, pair in groups.items():
        if len(pair) != 2 or {case['form'] for case in pair} != {'single', 'historical'}:
            raise ValueError('normal-suite semantic case requires exactly one single and one historical partner')
        first, second = pair
        document_id = first.get('document_id')
        if not isinstance(document_id, str) or not document_id:
            raise ValueError('normal-suite document identity missing')
        if any(first.get(field) != second.get(field) for field in common_fields):
            raise ValueError('normal-suite partners differ in document/blueprint/truth/question identity')
        if document_of(first) != document_of(second) or first['truth'] != second['truth']:
            raise ValueError('normal-suite partners differ in actual document or truth')
        if document_id in document_owners and document_owners[document_id] != semantic:
            raise ValueError('document ID reused for incompatible semantic cases')
        document_owners[document_id] = semantic


def make_plan(cases, costs, budget_seconds):
    """Costs must come from a predeclared pilot, never correctness outcomes."""
    validate_normal_suite_pairs(cases)
    if budget_seconds <= 0 or costs.get('cold_case_seconds', 0) <= 0:
        raise ValueError('positive budget and conservative fallback pilot cost required')
    scheduled, skipped, elapsed = [], [], 0.0
    budget_exhausted = False
    rows = sorted(profiles(), key=lambda profile: (profile['priority'], profile['id']))
    balanced_cases = sorted(cases, key=lambda case: (case['case_index'], TASKS.index(case['task']),
                                                     case['context_cap'], case['document_id'], case['form']))
    documents = defaultdict(list)
    for case in balanced_cases:
        documents[(case['case_index'], case['task'], case['context_cap'], case['document_id'])].append(case)
    # Cost-only pruning keeps both forms, both denominators and the shared
    # reference paired. Seed-major ordering spreads any reduced N over tasks/caps.
    for priority in (1, 2, 3, 4):
        for paired_cases in documents.values():
            bundle = []
            for case in paired_cases:
                for profile in rows:
                    if profile['priority'] != priority or case['context_cap'] != profile['context_cap']:
                        continue
                    cost = costs.get(profile['id'], {}).get(case['task'] + '/' + case['form'], costs['cold_case_seconds'])
                    if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
                        raise ValueError('invalid pilot cost')
                    bundle.append({'case_id': case['id'], 'profile_id': profile['id'], 'priority': priority, 'estimated_seconds': cost})
            bundle_cost = sum(row['estimated_seconds'] for row in bundle)
            if not bundle:
                continue
            if not budget_exhausted and elapsed + bundle_cost <= budget_seconds:
                scheduled.extend(bundle)
                elapsed += bundle_cost
            else:
                budget_exhausted = True
                for row in bundle:
                    row['reason'] = 'skipped-by-budget (pilot cost; no correctness selection)'
                skipped.extend(bundle)
    case_order = {case['id']: index for index, case in enumerate(balanced_cases)}
    scheduled.sort(key=lambda row: (row['priority'], row['profile_id'], case_order[row['case_id']]))
    return {'version': VERSION, 'profiles': rows, 'scheduled': scheduled, 'skipped_by_budget': skipped,
            'estimated_seconds': elapsed, 'budget_seconds': budget_seconds, 'pilot_costs': costs,
            'p4_status': 'informational-same-requested-token-counts'}


def append_event(path, event):
    with Path(path).open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(event).decode('utf-8') + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def production_source_sha():
    root = Path(__file__).resolve().parents[2]
    files = {path.relative_to(root).as_posix(): sha_file(path)
             for owner in ('src', 'include', 'apps') for path in (root / owner).rglob('*')
             if path.is_file() and path.suffix in ('.h', '.hpp', '.cuh', '.cpp', '.cu', '.c', '.cmake', '.json', '.js', '.ts', '.html', '.css')}
    return sha_bytes(canonical(files))


def check_production_identity(identity):
    if identity.get('source_tree_sha256') != production_source_sha():
        raise ValueError('production source tree identity missing/differs')
    files = identity.get('files', {})
    if not files:
        raise ValueError('frozen production binary/DLL file identities required')
    for path, digest in files.items():
        if sha_file(path) != digest:
            raise ValueError('production file identity changed: ' + path)


def read_raw(path):
    if not Path(path).exists():
        return {}
    rows = {}
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if index != len(lines)-1:
                raise ValueError('corrupt raw output before final line')
            break
        if row['case_id'] in rows:
            raise ValueError('duplicate raw output case identity')
        rows[row['case_id']] = row
    return rows


def measurement_result(case, profile, measured, logs, identity, raw_path, log_path, error=None):
    result = {'run_id': profile['id'] + '/' + case['id'], 'case_id': case['id'], 'profile_id': profile['id'],
              'input_sha256': case['input_sha256'], 'truth_sha256': case['truth_sha256'],
              'raw_output_path': str(raw_path), 'raw_log_path': str(log_path),
              'source_binary_identity': identity, 'profile': profile,
              'infrastructure_status': 'error', 'infrastructure_error': error}
    if measured and measured.get('infrastructure_status') == 'ok':
        metrics = measured['metrics']
        validation = validate(case, measured['content'])
        count_ok = metrics['prompt_tokens'] == case['prompt_tokens']
        summary_length_ok = case['task'] != 'summary' or metrics['generated_tokens'] == 1024
        turn_logs = logs.get(case['id'], [])
        log_ok = profile['mode'] != 'kvmem' or (turn_logs and all(log['denominator'] == profile['denominator'] for log in turn_logs))
        metrics['actual_view_tokens'] = (turn_logs[-1]['actual_view_tokens'] if turn_logs else metrics['resolved_main_kv_capacity_tokens'])
        metrics['actual_view_source'] = 'kvmem-turn' if turn_logs else 'public resolved main KV capacity'
        result.update(content=measured['content'], reasoning=measured['reasoning'], metrics=metrics,
                      validation=validation, kvmem_turn_logs=turn_logs,
                      infrastructure_status='ok' if count_ok and summary_length_ok and log_ok else 'error',
                      infrastructure_error=None if count_ok and summary_length_ok and log_ok else 'count/fixed-summary-length/denominator-log contract mismatch')
    elif measured:
        result['infrastructure_error'] = measured.get('error')
    return result


def recover_run(destination):
    from owned_process import OwnedTreeFatalError, process_alive
    dest = Path(destination)
    ledger = dest / 'events.jsonl'
    events = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
    if any(event['event'] == 'fatal-owned-tree' for event in events):
        raise OwnedTreeFatalError('fatal ownership/drain incident requires external inspection; automatic recovery/resume is forbidden')
    lock = dest / 'exclusive-run.lock'
    if lock.exists() and process_alive(int(lock.read_text(encoding='utf-8'))):
        raise ValueError('orchestrator still alive; recovery forbidden')
    identity = json.loads((dest / 'run-identity.json').read_text(encoding='utf-8'))
    if identity['source_sha256'] != source_identity():
        raise ValueError('recovery runner/rubric source identity differs')
    check_production_identity(identity['production_identity'])
    finished = {event['run_id'] for event in events if event['event'] == 'finished'}
    started = {event['run_id'] for event in events if event['event'] == 'started'}
    for event in events:
        if event['event'] != 'batch':
            continue
        batch = Path(event['batch'])
        if sha_file(batch / 'request.json') != event['request_sha256']:
            raise ValueError('recovery batch request SHA differs')
        request = json.loads((batch / 'request.json').read_text(encoding='utf-8'))
        raw = read_raw(batch / 'raw.jsonl')
        logs = sparse_logs(batch / 'stderr.txt') if (batch / 'stderr.txt').exists() else {}
        for case in request['cases']:
            profile = request['profile']
            run_id = profile['id'] + '/' + case['id']
            if run_id in finished or run_id not in started:
                continue
            audit_case(case)
            result = measurement_result(case, profile, raw.get(case['id']), logs, identity,
                                        batch / 'raw.jsonl', batch / 'stderr.txt',
                                        'interrupted batch: no completed output; not retried')
            append_event(ledger, {'event': 'finished', 'run_id': run_id, 'result': result, 'recovered': True})
    append_event(ledger, {'event': 'recovery-complete', 'no_inference_retried': True})
    if lock.exists():
        lock.unlink()


def sparse_logs(path):
    by_case, current = defaultdict(list), None
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('[quality-case-begin] '):
            current = line.split(' ', 1)[1]
        elif line.startswith('[quality-case-end] '):
            current = None
        elif line.startswith('[kvmem-turn] ') and current:
            row = json.loads(line.split(' ', 1)[1])
            row['gpu_queue_drain_wait_ms'] = row['capture_wait_ms']
            row['gpu_queue_drain_wait_is_prefill_subset'] = True
            by_case[current].append(row)
    return by_case


def report(manifest, cases, plan, results):
    lookup = {(r['case_id'], r['profile_id']): r for r in results}
    groups = defaultdict(list)
    case_map = {c['id']: c for c in cases}
    profile_map = {p['id']: p for p in plan['profiles']}
    for row in plan['scheduled']:
        case = case_map[row['case_id']]
        key = (case['task'], case['context_cap'], case['form'], row['profile_id'])
        groups[key].append(lookup.get((row['case_id'], row['profile_id'])))
        groups[(case['task'], case['context_cap'], 'combined', row['profile_id'])].append(lookup.get((row['case_id'], row['profile_id'])))
    summaries, paired, denominators = [], [], []
    for key, rows in sorted(groups.items()):
        completed = [r for r in rows if r and r['infrastructure_status'] == 'ok']
        passed = sum(r['validation']['quality_pass'] for r in completed)
        summary = dict(zip(('task', 'context_cap', 'form', 'profile_id'), key))
        summary.update(planned=len(rows), infrastructure_ok=len(completed), quality_pass=passed,
                       quality_success_rate=passed/len(completed) if completed else None,
                       strict_plan_success_rate=passed/len(rows),
                       unavailable_or_infrastructure_failed=len(rows)-len(completed))
        summary['validation_failure_reasons'] = dict(Counter(r['validation']['reason'] for r in completed if not r['validation']['quality_pass']))
        drafted = sum(r['metrics']['mtp_drafted_tokens'] for r in completed)
        accepted = sum(r['metrics']['mtp_accepted_tokens'] for r in completed)
        decode_seconds = sum(r['metrics']['decode_seconds'] for r in completed)
        summary.update(mtp_drafted_tokens=drafted, mtp_accepted_tokens=accepted,
                       mtp_acceptance=accepted/drafted if drafted else None,
                       decode_committed_tokens_per_second=sum(r['metrics']['committed_decode_tokens'] for r in completed)/decode_seconds if decode_seconds else None,
                       reused_prompt_tokens=sum(r['metrics']['reused_prompt_tokens'] for r in completed),
                       computed_prefill_tokens=sum(r['metrics']['computed_prefill_tokens'] for r in completed))
        if key[0] == 'summary':
            summary['mean_exact_fact_coverage'] = sum(r['validation']['fact_coverage_fraction'] for r in completed)/len(completed) if completed else None
            summary['all_ten_completion_rate'] = summary['quality_success_rate']
            summary['exact_1024_output_count'] = sum(r['metrics']['generated_tokens'] == 1024 for r in completed)
        summaries.append(summary)
    for profile in plan['profiles']:
        if not profile['reference_id']:
            continue
        for task in TASKS:
            for form in ('single', 'historical'):
                shared = [c for c in cases if c['task'] == task and c['form'] == form and c['context_cap'] == profile['context_cap']]
                pairs = [(lookup.get((c['id'], profile['id'])), lookup.get((c['id'], profile['reference_id']))) for c in shared]
                pairs = [(a, b) for a, b in pairs if a and b and a['infrastructure_status'] == b['infrastructure_status'] == 'ok']
                paired.append({'task': task, 'context_cap': profile['context_cap'], 'form': form,
                               'profile_id': profile['id'], 'reference_id': profile['reference_id'],
                               'paired_cases': len(pairs), 'paired_success_gap': sum(int(a['validation']['quality_pass'])-int(b['validation']['quality_pass']) for a, b in pairs)/len(pairs) if pairs else None})
                if profile['denominator'] == 'kept-band':
                    alternate = profile['id'].replace('kept-band', 'all-committed')
                    diffs = [(lookup.get((c['id'], profile['id'])), lookup.get((c['id'], alternate))) for c in shared]
                    diffs = [(a, b) for a, b in diffs if a and b and a['infrastructure_status'] == b['infrastructure_status'] == 'ok']
                    denominators.append({'task': task, 'context_cap': profile['context_cap'], 'form': form,
                                         'kept_profile_id': profile['id'], 'all_profile_id': alternate,
                                         'paired_cases': len(diffs), 'all_minus_kept_success_gap': sum(int(b['validation']['quality_pass'])-int(a['validation']['quality_pass']) for a, b in diffs)/len(diffs) if diffs else None})
    return {'version': VERSION, 'synthetic_only': True, 'real_corpus_acceptance_claimed': False,
            'validation_scope': {'retrieval': 'Strict chosen JSON structure and document-order constraints; semantic correctness alone may not pass.',
                                 'summary': 'Exact anchored entity/field/value coverage; correct paraphrases can be undercounted.',
                                 'failure_attribution': 'A malformed-format or exact-match failure does not by itself establish a memory or selection failure. Inspect preserved reason and raw output.'},
            'frozen_identity': manifest['identity'], 'groups': summaries, 'paired_reference_gaps': paired,
            'denominator_differences': denominators, 'skipped_by_budget': plan['skipped_by_budget'],
            'p4_status': plan['p4_status']}


def run_suite(frozen, plan_path, runner, model, destination, timeout):
    from owned_process import OwnedTreeFatalError, run_owned
    manifest, cases = load_frozen(frozen)
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan['version'] != VERSION or plan.get('frozen_manifest_sha256') != sha_file(Path(frozen) / 'manifest.json'):
        raise ValueError('plan frozen identity missing/differs')
    expected_plan = make_plan(cases, plan['pilot_costs'], plan['budget_seconds'])
    for key in expected_plan:
        if plan[key] != expected_plan[key]:
            raise ValueError('frozen priority/budget plan differs from cost-only selection')
    identity = {'manifest_sha256': sha_file(Path(frozen) / 'manifest.json'),
                'plan_sha256': sha_file(plan_path), 'binary_sha256': sha_file(runner),
                'model_sha256': sha_file(model), 'source_sha256': source_identity(),
                'production_identity': manifest['identity'].get('production_identity')}
    if identity['model_sha256'] != manifest['identity']['tokenizer_artifact_sha256'] or identity['binary_sha256'] != manifest['identity']['counter_binary_sha256']:
        raise ValueError('runner/model identity changed after freeze')
    check_production_identity(identity['production_identity'] or {})
    model_stat = (Path(model).stat().st_size, Path(model).stat().st_mtime_ns)
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    lock = dest / 'exclusive-run.lock'
    with lock.open('x', encoding='utf-8') as out:
        out.write(str(os.getpid()))
    # Leave lock after an interrupted orchestrator: explicit recovery is required.
    identity_path = dest / 'run-identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('resume source/input/runner/production identity differs')
    else:
        write_new(identity_path, identity)
    ledger = dest / 'events.jsonl'
    events = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    if any(event['event'] == 'fatal-owned-tree' for event in events):
        raise OwnedTreeFatalError('fatal ownership/drain incident recorded; no automatic resume or further dispatch')
    started = {event['run_id'] for event in events if event['event'] == 'started'}
    final = {event['run_id']: event['result'] for event in events if event['event'] == 'finished'}
    unresolved = started - final.keys()
    if unresolved:
        raise ValueError('started rows without finished output: inspect preserved raw batch; no automatic inference retry: ' + ','.join(sorted(unresolved)))
    profiles_by_id = {p['id']: p for p in plan['profiles']}
    cases_by_id = {c['id']: c for c in cases}
    # Batch contiguous same-profile cases to keep normal public Engine state.
    cursor = 0
    while cursor < len(plan['scheduled']):
        profile_id = plan['scheduled'][cursor]['profile_id']
        rows = []
        while cursor < len(plan['scheduled']) and plan['scheduled'][cursor]['profile_id'] == profile_id:
            row = plan['scheduled'][cursor]
            cursor += 1
            run_id = profile_id + '/' + row['case_id']
            if run_id not in final:
                rows.append((run_id, row))
        if not rows:
            continue
        batch_number = len([event for event in events if event['event'] == 'batch'])
        batch_dir = dest / f'batch-{batch_number:04d}'
        batch_dir.mkdir()
        profile = profiles_by_id[profile_id]
        if source_identity() != identity['source_sha256'] or sha_file(runner) != identity['binary_sha256']:
            raise ValueError('runner source/binary changed before batch dispatch')
        if (Path(model).stat().st_size, Path(model).stat().st_mtime_ns) != model_stat:
            raise ValueError('model metadata changed after SHA verification')
        check_production_identity(identity['production_identity'])
        request = {'operation': 'generate', 'model': str(Path(model).resolve()), 'profile': profile,
                   'policies': POLICIES, 'cases': [cases_by_id[row['case_id']] for _, row in rows]}
        write_new(batch_dir / 'request.json', request)
        append_event(ledger, {'event': 'batch', 'batch': str(batch_dir), 'profile_id': profile_id,
                              'request_sha256': sha_file(batch_dir / 'request.json')})
        events.append({'event': 'batch'})
        for run_id, _ in rows:
            append_event(ledger, {'event': 'started', 'run_id': run_id})
        environment = os.environ.copy()
        for name in list(environment):
            if name.startswith('NINFER_STAGE') or name.startswith('NINFER_KVMEM_'):
                environment.pop(name)
        if profile['denominator'] == 'all-committed':
            environment['NINFER_KVMEM_SCORE_ALL_PAGES'] = '1'
        error = None
        try:
            run_owned([str(runner), str(batch_dir / 'request.json'), str(batch_dir / 'raw.jsonl')],
                      environment=environment, stdout=batch_dir / 'stdout.txt', stderr=batch_dir / 'stderr.txt', timeout=timeout)
        except OwnedTreeFatalError as exc:
            append_event(ledger, {'event': 'fatal-owned-tree', 'batch': str(batch_dir),
                                  'profile_id': profile_id, 'error': str(exc),
                                  'child_pid': exc.child_pid, 'job_handle': exc.job_handle,
                                  'ownership_status': 'unresolved', 'further_dispatch_forbidden': True})
            raise  # Preserve lock, raw files, and unfinished rows; no normal report.
        except Exception as exc:
            error = str(exc)
        raw_path = batch_dir / 'raw.jsonl'
        raw = read_raw(raw_path)
        logs = sparse_logs(batch_dir / 'stderr.txt')
        for run_id, row in rows:
            case = cases_by_id[row['case_id']]
            result = measurement_result(case, profile, raw.get(case['id']), logs, identity,
                                        raw_path, batch_dir / 'stderr.txt', error)
            append_event(ledger, {'event': 'finished', 'run_id': run_id, 'result': result})
            final[run_id] = result
    output = report(manifest, cases, plan, list(final.values()))
    report_path = dest / ('report-' + sha_file(ledger)[:16] + '.json')
    if report_path.exists():
        if json.loads(report_path.read_text(encoding='utf-8')) != output:
            raise ValueError('existing checkpoint report differs')
    else:
        write_new(report_path, output)
    lock.unlink()  # Exact owned lock file only; no recursive deletion.
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    generate = sub.add_parser('draft')
    generate.add_argument('--output', required=True)
    generate.add_argument('--contexts', nargs='+', type=int, default=[131072, 262144])
    generate.add_argument('--cases', type=int, default=5, help='paired documents; each yields one case per form (default total 10)')
    generate.add_argument('--filler-lines', type=int, default=1000)
    generate.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    pilot = sub.add_parser('reuse-pilot-draft')
    pilot.add_argument('--output', required=True)
    pilot.add_argument('--context', type=int, default=32768)
    fit = sub.add_parser('fit-request')
    fit.add_argument('--draft', required=True)
    fit.add_argument('--model', required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--context', required=True, type=int)
    freeze_cmd = sub.add_parser('freeze')
    freeze_cmd.add_argument('--counted', nargs='+', required=True)
    freeze_cmd.add_argument('--identity', required=True)
    freeze_cmd.add_argument('--output', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--frozen', required=True)
    plan_cmd = sub.add_parser('plan')
    plan_cmd.add_argument('--frozen', required=True)
    plan_cmd.add_argument('--costs', required=True)
    plan_cmd.add_argument('--budget-seconds', required=True, type=float)
    plan_cmd.add_argument('--output', required=True)
    run = sub.add_parser('run')
    run.add_argument('--frozen', required=True)
    run.add_argument('--plan', required=True)
    run.add_argument('--runner', required=True)
    run.add_argument('--model', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--timeout', type=float, default=86400)
    recover = sub.add_parser('recover')
    recover.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'reuse-pilot-draft':
        cases = []
        base = make_cases('needle', args.context, 1, 80)[1]
        for index in range(3):
            case = json.loads(canonical(base))
            _, question, truth = task_data('needle', 0, index)
            case.update(id=f'needle-reuse-pilot-q{index}', query_index=index,
                        question=question, truth=json.loads(canonical(truth)))
            case['messages'][2]['content'] = question
            stamp_case(case)
            audit_case(case)
            cases.append(case)
        write_new(args.output, {'version': VERSION, 'policies': POLICIES, 'cases': cases,
                                'source_sha256': source_identity(), 'pilot_only': True})
        print('three frozen historical queries share one exact document and acknowledgement')
    elif args.command == 'draft':
        if args.cases < 1 or args.filler_lines < 80:
            parser.error('positive cases and at least 80 filler lines required')
        cases = [case for cap in args.contexts for task in args.tasks for case in make_cases(task, cap, args.cases, args.filler_lines)]
        for case in cases:
            audit_case(case)
        write_new(args.output, {'version': VERSION, 'policies': POLICIES, 'cases': cases, 'source_sha256': source_identity()})
        print(f'{len(cases)} deterministic draft cases; actual tokenizer fitting required')
    elif args.command == 'fit-request':
        draft = json.loads(Path(args.draft).read_text(encoding='utf-8'))
        if draft['version'] != VERSION or draft['policies'] != POLICIES or draft['source_sha256'] != source_identity():
            raise ValueError('draft recipe/rubric/source identity differs before fitting')
        cases = [case for case in draft['cases'] if case['context_cap'] == args.context]
        for case in cases:
            audit_case(case)
        profile = {'context_cap': args.context, 'mode': 'kvmem', 'dtype': 'int8',
                   'requested_view_tokens': 32768, 'denominator': 'kept-band'}
        write_new(args.output, {'operation': 'fit', 'model': args.model, 'profile': profile,
                                'policies': POLICIES, 'cases': cases})
    elif args.command == 'freeze':
        cases = []
        for path in args.counted:
            for line in Path(path).read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if row['infrastructure_status'] != 'ok':
                    raise ValueError('counted fixture has infrastructure failure')
                case = row['case']
                original_blueprint = case['blueprint_sha256']
                stamp_case(case)
                if case['blueprint_sha256'] != original_blueprint:
                    raise ValueError('count fitting changed injection/truth/question/policy blueprint')
                cases.append(case)
        manifest = freeze(cases, args.output, json.loads(Path(args.identity).read_text(encoding='utf-8')))
        print(f"frozen {manifest['case_count']} cases")
    elif args.command == 'audit':
        _, cases = load_frozen(args.frozen)
        print(f'audited {len(cases)} frozen cases')
    elif args.command == 'plan':
        _, cases = load_frozen(args.frozen)
        plan = make_plan(cases, json.loads(Path(args.costs).read_text(encoding='utf-8')), args.budget_seconds)
        plan['frozen_manifest_sha256'] = sha_file(Path(args.frozen) / 'manifest.json')
        write_new(args.output, plan)
        print(f"scheduled {len(plan['scheduled'])}; skipped {len(plan['skipped_by_budget'])}")
    elif args.command == 'recover':
        recover_run(args.output)
        print('checkpoint reconciled without repeating any started inference')
    else:
        output = run_suite(args.frozen, args.plan, args.runner, args.model, args.output, args.timeout)
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
