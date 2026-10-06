"""CPU-only contract tests for the frozen local synthetic quality suite."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'quality'))
import kvmem_quality as q


class QualityContracts(unittest.TestCase):
    def fixtures(self, task):
        return q.make_cases(task, 4096, 2, 80)

    def answer(self, case):
        truth = case['truth']
        if case['task'] == 'summary':
            return '\n'.join(' | '.join(fact) for fact in truth['facts'])
        return json.dumps(truth['answer'])

    def test_determinism_and_independent_audit(self):
        for task in q.TASKS:
            cases = self.fixtures(task)
            self.assertEqual(cases, self.fixtures(task))
            for case in cases:
                q.audit_case(case)
                self.assertTrue(q.validate(case, self.answer(case))['quality_pass'])
            self.assertEqual(cases[0]['document_sha256'], cases[1]['document_sha256'])
            self.assertEqual(cases[0]['current_input_message'], 0)
            self.assertEqual(cases[1]['current_input_message'], 2)

    def test_wrong_distractor_missing_duplicate_and_extra(self):
        one = self.fixtures('needle')[0]
        bad = {'reference': one['truth']['entries'][1][1]}
        self.assertFalse(q.validate(one, json.dumps(bad))['quality_pass'])
        repeated = self.fixtures('repeat')[0]
        values = repeated['truth']['answer']['references']
        for bad_values in (values[:-1], values + values[:1], values[:1] * 4):
            self.assertFalse(q.validate(repeated, json.dumps({'references': bad_values}))['quality_pass'])
        self.assertFalse(q.validate(one, self.answer(one) + ' explanation')['quality_pass'])
        self.assertFalse(q.validate(one, '{"reference":"00000000","reference":' + json.dumps(one['truth']['answer']['reference']) + '}')['quality_pass'])

    def test_wrong_graph_rank_and_fact_link(self):
        chain = self.fixtures('chain')[0]
        self.assertFalse(q.validate(chain, json.dumps({'variables': ['X1', 'X2', 'X3', 'X4']}))['quality_pass'])
        rank = self.fixtures('frequency')[0]
        self.assertFalse(q.validate(rank, json.dumps({'top3': list(reversed(rank['truth']['answer']['top3']))}))['quality_pass'])
        summary = self.fixtures('summary')[0]
        facts = copy.deepcopy(summary['truth']['facts'])
        facts[0][2], facts[1][2] = facts[1][2], facts[0][2]
        result = q.validate(summary, '\n'.join(' | '.join(f) for f in facts))
        self.assertEqual(result['fact_coverage_count'], 8)
        self.assertFalse(result['quality_pass'])

    def test_audit_mutations(self):
        for task in q.TASKS:
            case = copy.deepcopy(self.fixtures(task)[0])
            document = q.document_of(case)
            if task in ('needle', 'multi', 'repeat'):
                document = document.replace(case['truth']['entries'][0][1], '99999999', 1)
            elif task == 'chain':
                document = document.replace('X5=X4', 'X5=Y4', 1)
            elif task == 'frequency':
                document += '\n' + case['truth']['markers'][0]
            else:
                document = document.replace(case['truth']['facts'][0][2], 'WRONG', 1)
            q.set_document(case, document)
            q.stamp_case(case)  # Updating hashes must not defeat the input audit.
            with self.assertRaises(ValueError):
                q.audit_case(case)

    def test_corrupt_sha_and_answer_leak(self):
        case = copy.deepcopy(self.fixtures('needle')[0])
        case['messages'][0]['content'] += ' changed'
        with self.assertRaises(ValueError):
            q.audit_case(case)
        case = copy.deepcopy(self.fixtures('needle')[0])
        case['messages'][0]['content'] += '\n' + case['truth']['answer']['reference']
        q.stamp_case(case)
        with self.assertRaises(ValueError):
            q.audit_case(case)

    def test_frozen_roundtrip_and_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'freeze'
            cases = self.fixtures('needle')
            for case in cases:
                case['prompt_tokens'] = 3000
            q.freeze(cases, path, {'tokenizer_artifact_sha256': 'a' * 64, 'counter_binary_sha256': 'b' * 64})
            manifest, loaded = q.load_frozen(path)
            self.assertEqual(cases, loaded)
            self.assertEqual(manifest['case_count'], 4)
            with self.assertRaises(FileExistsError):
                q.freeze(cases, path, {})
            (path / 'cases' / (cases[0]['id'] + '.json')).write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                q.load_frozen(path)

    def test_freeze_rejects_unsafe_case_ids_before_any_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            invalid_ids = ('../../escaped-case', '/absolute-case', str(base / 'absolute-case'),
                           'C:\\escaped-case', 'C:escaped-case', 'nested/case', 'nested\\case',
                           '..', '', 'case\x00name', 'CON', 'NUL', 'COM1', None, 7, ['case'])
            for index, invalid_id in enumerate(invalid_ids):
                with self.subTest(case_id=invalid_id):
                    destination = base / f'freeze-{index}'
                    valid = copy.deepcopy(self.fixtures('needle')[0])
                    invalid = copy.deepcopy(self.fixtures('needle')[1])
                    valid['prompt_tokens'] = invalid['prompt_tokens'] = 3000
                    invalid['id'] = invalid_id
                    with self.assertRaises(ValueError):
                        q.freeze([valid, invalid], destination, {'tokenizer_artifact_sha256': 'a' * 64,
                                                               'counter_binary_sha256': 'b' * 64})
                    self.assertFalse(destination.exists())
                    self.assertEqual(list(base.iterdir()), [])
            pair = copy.deepcopy(self.fixtures('needle')[:2])
            pair[0]['id'], pair[1]['id'] = 'same-case', 'SAME-CASE'
            for case in pair:
                case['prompt_tokens'] = 3000
            with self.assertRaises(ValueError):
                q.freeze(pair, base / 'case-collision', {'tokenizer_artifact_sha256': 'a' * 64,
                                                       'counter_binary_sha256': 'b' * 64})
            self.assertEqual(list(base.iterdir()), [])

    def test_unknown_form_is_rejected_by_audit(self):
        case = copy.deepcopy(self.fixtures('needle')[1])
        case['form'] = 'unknown'
        with self.assertRaises(ValueError):
            q.audit_case(case)

    def test_normal_plan_requires_complete_identical_partners(self):
        pair = q.make_cases('needle', 262144, 1, 80)
        costs = {'cold_case_seconds': 1}
        mutations = [pair[:1], [pair[0], pair[0]], pair + [copy.deepcopy(pair[1])]]
        different_document = copy.deepcopy(pair)
        different_document[1]['recipe'] = copy.deepcopy(different_document[1]['recipe'])
        different_document[1]['recipe']['filler_lines'] = 81
        q.set_document(different_document[1], q.render_document(different_document[1]['recipe']))
        q.stamp_case(different_document[1])
        q.audit_case(different_document[0])
        q.audit_case(different_document[1])
        self.assertNotEqual(different_document[0]['document_sha256'], different_document[1]['document_sha256'])
        mutations.append(different_document)
        different_query = copy.deepcopy(pair)
        _, question, truth = q.task_data('needle', 0, 1)
        different_query[1].update(query_index=1, question=question, truth=json.loads(q.canonical(truth)))
        different_query[1]['messages'][2]['content'] = question
        q.stamp_case(different_query[1])
        q.audit_case(different_query[1])
        self.assertEqual(different_query[0]['document_sha256'], different_query[1]['document_sha256'])
        mutations.append(different_query)
        for field, value in (('document_id', 'different-document'), ('context_cap', 131072),
                             ('question', 'changed question'), ('case_index', 1), ('task', 'multi')):
            bad = copy.deepcopy(pair)
            bad[1][field] = value
            mutations.append(bad)
        duplicate_form = copy.deepcopy(pair)
        duplicate_form[1] = copy.deepcopy(pair[0])
        duplicate_form[1]['id'] += '-duplicate-form'
        mutations.append(duplicate_form)
        for bad in mutations:
            with self.subTest(ids=[c['id'] for c in bad]):
                with self.assertRaises(ValueError):
                    q.make_plan(bad, costs, 1000)
        plan = q.make_plan(pair, costs, 1000)
        self.assertEqual(len(plan['scheduled']), 20)  # ten ctx262K profiles, two forms.

    def test_isolated_historical_pilot_freeze_remains_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            pilot = q.make_cases('needle', 32768, 1, 80)[1]
            pilot['id'] = 'isolated-historical-cost-pilot'
            pilot['prompt_tokens'] = 3000
            destination = Path(folder) / 'pilot'
            q.freeze([pilot], destination, {'tokenizer_artifact_sha256': 'a' * 64,
                                          'counter_binary_sha256': 'b' * 64})
            _, loaded = q.load_frozen(destination)
            self.assertEqual(loaded, [pilot])
            queries = []
            for index in range(3):
                case = copy.deepcopy(pilot)
                _, question, truth = q.task_data('needle', 0, index)
                case.update(id=f'reuse-pilot-q{index}', query_index=index,
                            question=question, truth=json.loads(q.canonical(truth)))
                case['messages'][2]['content'] = question
                q.stamp_case(case)
                queries.append(case)
            q.freeze(queries, Path(folder) / 'reuse-pilot', {'tokenizer_artifact_sha256': 'a' * 64,
                                                          'counter_binary_sha256': 'b' * 64})
            _, loaded = q.load_frozen(Path(folder) / 'reuse-pilot')
            self.assertEqual(loaded, queries)
            with self.assertRaises(ValueError):
                q.make_plan(queries, {'cold_case_seconds': 1}, 1000)

    def test_all_120_normal_cases_remain_paired_and_plan_960_requests(self):
        cases = [case for cap in (131072, 262144) for task in q.TASKS
                 for case in q.make_cases(task, cap, 5, 80)]
        plan = q.make_plan(cases, {'cold_case_seconds': 1}, 1000000)
        self.assertEqual(len(cases), 120)
        self.assertEqual(len(plan['scheduled']), 960)
        self.assertEqual(len({(r['case_id'], r['profile_id']) for r in plan['scheduled']}), 960)
        self.assertEqual(plan['skipped_by_budget'], [])

    def test_budget_priority_and_reference_deduplication(self):
        cases = sum((q.make_cases('needle', cap, 1, 80) for cap in (131072, 262144)), [])
        plan = q.make_plan(cases, {'cold_case_seconds': 1}, 1000)
        identities = [(r['case_id'], r['profile_id']) for r in plan['scheduled']]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertTrue(any(r['priority'] == 4 for r in plan['scheduled']))
        self.assertEqual({p['requested_view_tokens'] for p in plan['profiles'] if p['priority'] == 4}, {32768, 131072})
        pruned = q.make_plan(cases, {'cold_case_seconds': 1}, 6)
        self.assertTrue(pruned['skipped_by_budget'])
        self.assertEqual(pruned['scheduled'][0]['priority'], 1)

    def test_blueprint_identity_survives_neutral_fit(self):
        case = self.fixtures('needle')[0]
        before = case['blueprint_sha256']
        case['recipe'] = dict(case['recipe'], filler_lines=120)
        q.set_document(case, q.render_document(case['recipe']))
        q.stamp_case(case)
        q.audit_case(case)
        self.assertEqual(before, case['blueprint_sha256'])

    def test_raw_partial_final_line_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'raw.jsonl'
            path.write_text('{"case_id":"one"}\n{"case_id":', encoding='utf-8')
            self.assertEqual(list(q.read_raw(path)), ['one'])
            path.write_text('{"case_id":"one"}\n{"case_id":"one"}\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                q.read_raw(path)

    def test_summary_duplicate_links_and_unanchored_values_fail(self):
        case = self.fixtures('summary')[0]
        text = self.answer(case)
        self.assertFalse(q.validate(case, text + '\n' + text.splitlines()[0])['quality_pass'])
        loose = ' '.join(fact[2] for fact in case['truth']['facts'])
        self.assertEqual(q.validate(case, loose)['fact_coverage_count'], 0)

    def test_recovery_reconciles_partial_output_without_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            runner = base / 'production.bin'
            runner.write_bytes(b'fixed production')
            identity = {'source_sha256': q.source_identity(),
                        'production_identity': {'source_tree_sha256': q.production_source_sha(),
                                                'files': {str(runner): q.sha_file(runner)}}}
            q.write_new(base / 'run-identity.json', identity)
            cases = self.fixtures('needle')[:2]
            for case in cases:
                case['prompt_tokens'] = 3000
            profile = {'id': 'recovery-dense', 'mode': 'dense'}
            batch = base / 'batch-0000'
            q.write_new(batch / 'request.json', {'profile': profile, 'cases': cases})
            ledger = base / 'events.jsonl'
            q.append_event(ledger, {'event': 'batch', 'batch': str(batch), 'request_sha256': q.sha_file(batch / 'request.json')})
            for case in cases:
                q.append_event(ledger, {'event': 'started', 'run_id': profile['id'] + '/' + case['id']})
            metrics = {'prompt_tokens': 3000, 'generated_tokens': 10, 'resolved_main_kv_capacity_tokens': 4096}
            row = {'case_id': cases[0]['id'], 'infrastructure_status': 'ok', 'metrics': metrics,
                   'content': self.answer(cases[0]), 'reasoning': ''}
            (batch / 'raw.jsonl').write_text(json.dumps(row) + '\n{"partial":', encoding='utf-8')
            (batch / 'stderr.txt').write_text('', encoding='utf-8')
            with mock.patch('owned_process.run_owned', side_effect=AssertionError('recovery must never infer')):
                q.recover_run(base)
            events = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
            completed = [event['result'] for event in events if event['event'] == 'finished']
            self.assertEqual(len(completed), 2)
            self.assertEqual([r['infrastructure_status'] for r in completed], ['ok', 'error'])
            self.assertTrue(completed[0]['validation']['quality_pass'])

    def test_quality_failure_continues_and_resume_never_dispatches_again(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            model, runner = base / 'model.bin', base / 'runner.bin'
            model.write_bytes(b'model fixture identity')
            runner.write_bytes(b'runner fixture identity')
            identity = {'tokenizer_artifact_sha256': q.sha_file(model),
                        'counter_binary_sha256': q.sha_file(runner),
                        'production_identity': {'source_tree_sha256': q.production_source_sha(),
                                                'files': {str(runner): q.sha_file(runner)}}}
            cases = q.make_cases('needle', 131072, 1, 80)
            for case in cases:
                case['prompt_tokens'] = 3000
            frozen = base / 'frozen'
            q.freeze(cases, frozen, identity)
            plan = q.make_plan(cases, {'cold_case_seconds': 1}, 1000)
            plan['frozen_manifest_sha256'] = q.sha_file(frozen / 'manifest.json')
            plan_path = base / 'plan.json'
            q.write_new(plan_path, plan)
            dispatches = []
            def fake_run(command, **options):
                request = json.loads(Path(command[1]).read_text(encoding='utf-8'))
                profile = request['profile']
                dispatches.append(profile['id'])
                logs = []
                with Path(command[2]).open('x', encoding='utf-8') as output:
                    for case in request['cases']:
                        content = self.answer(case)
                        if len(dispatches) == 1:
                            content = '{"reference":"00000000"}'
                        metrics = {'prompt_tokens': 3000, 'generated_tokens': 10,
                                   'resolved_main_kv_capacity_tokens': profile['requested_view_tokens'],
                                   'mtp_drafted_tokens': 9, 'mtp_accepted_tokens': 6,
                                   'committed_decode_tokens': 9, 'decode_seconds': 1,
                                   'reused_prompt_tokens': 0, 'computed_prefill_tokens': 3000}
                        output.write(json.dumps({'case_id': case['id'], 'infrastructure_status': 'ok',
                                                 'metrics': metrics, 'content': content, 'reasoning': ''}) + '\n')
                        logs.append('[quality-case-begin] ' + case['id'])
                        if profile['mode'] == 'kvmem':
                            logs.append('[kvmem-turn] ' + json.dumps({'denominator': profile['denominator'],
                                       'capture_wait_ms': 2, 'actual_view_tokens': profile['requested_view_tokens']}))
                        logs.append('[quality-case-end] ' + case['id'])
                Path(options['stdout']).write_text('', encoding='utf-8')
                Path(options['stderr']).write_text('\n'.join(logs), encoding='utf-8')
            with mock.patch('owned_process.run_owned', side_effect=fake_run):
                output = q.run_suite(frozen, plan_path, runner, model, base / 'run', 10)
                self.assertEqual(len(dispatches), 6)
                self.assertTrue(any(group['quality_pass'] == 0 and group['infrastructure_ok'] == 1 for group in output['groups']))
                q.run_suite(frozen, plan_path, runner, model, base / 'run', 10)
                self.assertEqual(len(dispatches), 6)
                model.write_bytes(b'changed model')
                with self.assertRaises(ValueError):
                    q.run_suite(frozen, plan_path, runner, model, base / 'run', 10)

    def test_fatal_owned_tree_stops_dispatch_preserves_evidence_and_blocks_recovery(self):
        from owned_process import OwnedTreeFatalError
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            model, runner = base / 'model.bin', base / 'runner.bin'
            model.write_bytes(b'model identity')
            runner.write_bytes(b'runner identity')
            identity = {'tokenizer_artifact_sha256': q.sha_file(model),
                        'counter_binary_sha256': q.sha_file(runner),
                        'production_identity': {'source_tree_sha256': q.production_source_sha(),
                                                'files': {str(runner): q.sha_file(runner)}}}
            cases = q.make_cases('needle', 131072, 1, 80)
            for case in cases:
                case['prompt_tokens'] = 3000
            frozen = base / 'frozen'
            q.freeze(cases, frozen, identity)
            plan = q.make_plan(cases, {'cold_case_seconds': 1}, 1000)
            plan['frozen_manifest_sha256'] = q.sha_file(frozen / 'manifest.json')
            plan_path = base / 'plan.json'
            q.write_new(plan_path, plan)
            dispatches = []
            def fatal_dispatch(command, **options):
                dispatches.append(command)
                Path(options['stdout']).write_text('retained stdout', encoding='utf-8')
                Path(options['stderr']).write_text('retained ownership failure', encoding='utf-8')
                Path(command[2]).write_text('{"partial":', encoding='utf-8')
                raise OwnedTreeFatalError('injected unconfirmed JOB drain', child_pid=123, job_handle=456)
            destination = base / 'run'
            with mock.patch('owned_process.run_owned', side_effect=fatal_dispatch):
                with self.assertRaises(OwnedTreeFatalError):
                    q.run_suite(frozen, plan_path, runner, model, destination, 10)
                self.assertEqual(len(dispatches), 1)
                self.assertTrue((destination / 'exclusive-run.lock').exists())
                self.assertEqual(list(destination.glob('report-*.json')), [])
                self.assertEqual((destination / 'batch-0000' / 'raw.jsonl').read_text(encoding='utf-8'), '{"partial":')
                events = [json.loads(line) for line in (destination / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
                fatal = [event for event in events if event['event'] == 'fatal-owned-tree']
                self.assertEqual(len(fatal), 1)
                self.assertEqual(fatal[0]['child_pid'], 123)
                self.assertFalse(any(event['event'] == 'finished' for event in events))
                with self.assertRaises(OwnedTreeFatalError):
                    q.recover_run(destination)
                self.assertTrue((destination / 'exclusive-run.lock').exists())
                (destination / 'exclusive-run.lock').unlink()
                with self.assertRaises(OwnedTreeFatalError):
                    q.run_suite(frozen, plan_path, runner, model, destination, 10)
                self.assertEqual(len(dispatches), 1)

    def test_ordinary_drained_child_failure_remains_nonfatal(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            model, runner = base / 'model.bin', base / 'runner.bin'
            model.write_bytes(b'model identity')
            runner.write_bytes(b'runner identity')
            identity = {'tokenizer_artifact_sha256': q.sha_file(model),
                        'counter_binary_sha256': q.sha_file(runner),
                        'production_identity': {'source_tree_sha256': q.production_source_sha(),
                                                'files': {str(runner): q.sha_file(runner)}}}
            cases = q.make_cases('needle', 131072, 1, 80)
            for case in cases:
                case['prompt_tokens'] = 3000
            frozen = base / 'frozen'
            q.freeze(cases, frozen, identity)
            plan = q.make_plan(cases, {'cold_case_seconds': 1}, 1000)
            plan['frozen_manifest_sha256'] = q.sha_file(frozen / 'manifest.json')
            plan_path = base / 'plan.json'
            q.write_new(plan_path, plan)
            dispatches = []
            def ordinary_failure(command, **options):
                dispatches.append(command)
                Path(options['stdout']).write_text('', encoding='utf-8')
                Path(options['stderr']).write_text('ordinary child failure after confirmed drain', encoding='utf-8')
                raise subprocess.CalledProcessError(7, command)
            destination = base / 'run'
            with mock.patch('owned_process.run_owned', side_effect=ordinary_failure):
                report = q.run_suite(frozen, plan_path, runner, model, destination, 10)
            self.assertEqual(len(dispatches), 6)
            self.assertFalse((destination / 'exclusive-run.lock').exists())
            self.assertTrue(all(group['infrastructure_ok'] == 0 for group in report['groups']))


if __name__ == '__main__':
    unittest.main()
