"""Reproduce the bounded pilot synopsis from the archived text evidence.

No PyTorch, trained weights, NPZ data, simulation, or training is needed.
This checks archival hashes and summarizes measured outputs, not acceptance odds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def build(archive_path: Path):
    raw_archive = archive_path.read_bytes()
    archive = json.loads(gzip.decompress(raw_archive))
    inputs = {}

    def read(name):
        item = archive['files'][name]
        raw = item['utf8'].encode()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'archive checksum failed: {name}')
        inputs[name] = digest
        return json.loads(raw)

    campaign = read('campaign.json')
    source = read('source/summary.json')
    study = read('source/study.json')
    source_manifest = read('source/data/manifest.json')
    transfer = read('transfer/summary.json')
    literature = read('literature/summary.json')
    lit_campaign = read('literature/campaign.json')
    if campaign['status'] != 'complete' or any(x['status'] != 'complete' for x in campaign['steps'].values()):
        raise ValueError('campaign is not complete')
    if {archive['source_hash'], campaign['source_hash'], source['source_hash'], transfer['source_hash']} != {archive['source_hash']}:
        raise ValueError('source hashes disagree')
    # Recompute primary means from the actual parent/seed panel, not prose.
    cells = defaultdict(lambda: defaultdict(list))
    for row in source['records']:
        cells[(row['method'], row['mode'])][(row['parent_id'], row['seed'])].append(row['loss'])
    for (method, mode), values in cells.items():
        actual = mean(mean(v) for v in values.values())
        if abs(actual - source['means'][method][mode]) > 1e-12:
            raise ValueError(f'source record means disagree: {method}/{mode}')
    transfer_rows = [{key: result[key] for key in ('target', 'method', 'n_seeds', 'mean_selected_loss', 'crossed_parent_seed_contrasts')}
                     for result in transfer['results']]
    budgets = []
    for name in sorted(campaign['steps']):
        if not name.startswith('budget_'):
            continue
        report = read(name + '/report.json')
        manifest = read(name + '/manifest.json')
        config = manifest['frozen']['config']
        costs = Counter()
        for row in report['cost_ledger']:
            costs[row['role']] += row['attempted_objective_calls']
        budgets.append({'run': name, 'training_arm': name.split('_')[1], 'training_seed': int(name.rsplit('_', 1)[1]),
                        'optimizer_seeds': config['seeds'], 'budgets': config['budgets'],
                        'n_records': report['n_records'], 'n_parents': report['n_logical_parents'],
                        'checkpoint_sha256': manifest['frozen']['checkpoint_sha256'],
                        'curves': [{k: row[k] for k in ('method', 'budget', 'parent_mean_loss', 'requested_rows', 'completed_rows', 'censored_rows')}
                                   for row in report['curves']],
                        'paired_warm_minus_cold': report['paired_warm_minus_cold'],
                        'objective_calls': dict(costs), 'failed_calls': sum(row['failed_calls'] for row in report['cost_ledger']),
                        'break_even': report['break_even'],
                        'shared_offline_preparation_seconds': report['offline_cost']['total_seconds'],
                        'offline_cost_scope': report['offline_cost']['scope']})
    parents = Counter(source_manifest['splits'].values())
    records = Counter(row['split'] for row in source_manifest['records'])
    point_ci = lambda r: f"{r['mean_difference']:+.6f} [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}]"
    report = {'schema_version': 1, 'archive': archive_path.name,
              'archive_sha256': hashlib.sha256(raw_archive).hexdigest(),
              'source_hash': archive['source_hash'], 'campaign_config_hash': campaign['config_hash'],
              'input_sha256': inputs, 'status': campaign['status'], 'completed_steps': len(campaign['steps']),
              'observed_campaign_wall_seconds': campaign['observed_wall_seconds'],
              'scope': 'exploratory CPU pilot; negative and inconclusive outcomes retained; not a confirmatory efficacy or venue-acceptance claim',
              'source': {'parent_counts': dict(parents), 'record_counts': dict(records), 'training_seeds': study['config']['seeds'],
                         'model': study['config']['model'], 'training': study['config']['training'],
                         'n_training_fits_including_baselines_and_mechanisms': len(study['runs']),
                         'means': source['means'], 'acquisition_contrasts': source['acquisition_contrasts'],
                         'mechanism_contrasts': source['mechanism_contrasts'],
                         'offline_diagnostic_objective_calls': sum(row['offline_evaluation_diagnostics']['offline_objective_calls'] for row in source['costs'].values()),
                         'interpretation': 'POLICY does not improve on CONTROL, BANKEXT, or decoder-random convincingly; no isolated mechanism contrast is established'},
              'transfer': {'axes': transfer['transfer_axes'], 'global_baseline': transfer['global_baseline'],
                           'results': transfer_rows, 'costs': transfer['costs'],
                           'interpretation': 'unseen-family improvement over linear is entirely matched by source-global; unseen-runtime intervals do not establish robust improvement'},
              'budgets': budgets, 'literature': literature,
              'literature_config': lit_campaign['frozen']['config'],
              'archive_omissions': archive['omitted'],
              'separate_upstream_evidence': '../evidence_audit_2026-09-18/UPSTREAM_TRANSFER.md'}
    lines = ['# Bounded paper campaign: measured pilot results', '',
             'All **11 configured stages completed** on CPU. The controlled measurements are mostly negative or inconclusive: policy acquisition does not improve direct selection reliably, family transfer gains are matched by a fixed source-selected schedule, and the pilot establishes no amortization break-even. Successful execution validates the experimental workflow; it does not establish an A* main-track contribution or guarantee acceptance.', '',
             f"Frozen source fingerprint: `{archive['source_hash']}`. Campaign archive: `{archive_path.name}`; SHA-256 `{report['archive_sha256']}`. Observed campaign execution time was {campaign['observed_wall_seconds']:.3f} seconds; this is one CPU run, not a scaling benchmark.", '',
             '## Design and independent units', '',
             f"Source: **48 logical parents**, split {parents['train']} train / {parents['validation']} validation / {parents['test']} test; {len(source_manifest['records'])} records, 16 candidate schedules each. Logical sizes 3–4, physical sizes 3–6, synthetic spin-glass/weighted-MaxCut families, runtime 1 or 3. The summary encoder has width 32 and four proposals, at most 30 epochs with patience 8. Training seeds are 0, 1, 2. There are **33 fits**: three frozen baselines, twelve primary-arm fits, and eighteen mechanism continuations. The embedded historical `_scope` string incorrectly says 60 parents; the actual manifest and configuration specify 48.", '',
             'Each primary acquisition arm adds four schedules to each of 64 training records: 256 objective labels per arm/seed, 2,304 labels across BANKEXT, decoder-random, and POLICY. Validation/test banks stay frozen. Mechanism arms reuse acquired labels and hold the encoder fixed. Source evaluation adds 3,840 offline diagnostic objective calls; those are not deployment queries.', '',
             'Intervals below are descriptive crossed parent×training-seed bootstraps with 2,000 resamples, conditional on this dataset and recipe. They are unadjusted and not equivalence tests; three training seeds and eight test parents provide limited precision.', '',
             '## Four acquisition arms', '', '| Arm | Bank loss | Direct loss |', '|---|---:|---:|']
    for arm in ('control', 'bankext', 'decoder_random', 'policy'):
        row = source['means'][arm]
        lines.append(f"| {arm} | {row['bank']:.6f} | {row['direct']:.6f} |")
    lines += ['', 'Global fixed schedule: 0.709660; linear: 0.735915. Lower loss is better.', '',
              '| Direct-mode contrast | Difference and 95% crossed CI |', '|---|---|']
    for name, row in source['acquisition_contrasts'].items():
        lines.append(f'| {name} | {point_ci(row)} |')
    lines += ['', 'POLICY is worse than CONTROL on all three seed averages. Its descriptive interval only narrowly excludes zero; this is not a multiplicity-adjusted finding. Neither POLICY−BANKEXT nor POLICY−decoder-random excludes zero. These outcomes must remain visible rather than selecting the favourable historical aggregation campaign.', '',
              '| Frozen-backbone mechanism: acquired − original labels | Direct difference and 95% crossed CI |', '|---|---|']
    for name, modes in source['mechanism_contrasts'].items():
        lines.append(f"| {name} | {point_ci(modes['direct'])} |")
    lines += ['', 'All direct mechanism intervals contain zero. Critic/head bank-mode intervals reach zero; policy-only continuation leaves the bank predictions unchanged as expected. No causal repair of critic distribution shift or proposal generation is established by this pilot.', '',
              '## Transfer with source-selected global control', '',
              'CONTROL and POLICY checkpoints from all three training seeds were evaluated with frozen source normalizers. The global waveform was selected from eight source-validation parents, without target labels. Every target evaluation has eight logical parents: 16 source-held-out records, 16 unseen-family (`weak_field`) records, or eight unseen-runtime (`T=8`) records. Eighteen checkpoint-target evaluations completed with zero online simulator queries. This is synthetic family/runtime transfer, not topology or larger-qubit transfer.', '',
              '| Target | Checkpoint arm | Bank loss | Selector − source-global: difference and 95% crossed CI |', '|---|---|---:|---|']
    for row in transfer_rows:
        lines.append(f"| {row['target']} | {row['method']} | {row['mean_selected_loss']:.6f} | {point_ci(row['crossed_parent_seed_contrasts']['source_global'])} |")
    lines += ['', 'On unseen family, both learned selectors and source-global obtain 0.685273; the −0.050568 improvement over linear is therefore not evidence of added learned per-instance value. At runtime 8, CONTROL−linear is −0.022316 [−0.108558, +0.056430], and POLICY−linear is +0.035277 [−0.007527, +0.063986]. CONTROL’s favourable mean relative to global comes entirely from training seed 2 and its interval reaches zero. Transfer is not established as robust across training seeds.', '',
              'The separate upstream synthetic→Pegasus aggregates in commit `974207a` concern different checkpoints, instances, and sizes. They are audited in [UPSTREAM_TRANSFER.md](../evidence_audit_2026-09-18/UPSTREAM_TRANSFER.md) and are not pooled with this pilot.', '',
              '## Six budget studies', '',
              'One report per CONTROL/POLICY checkpoint × three training seeds. Each uses the **same six source-test records from three parents**, optimizer seeds 0 and 1, budgets 0/2/5/9, and one eight-bin family. Search seeds are not extra training seeds. Each search trajectory supplies nested budget prefixes; budget points and repeated cold-start runs across checkpoints are not independent replications.', '',
              '| Checkpoint | Direct at 0 calls | Sobol direct-warm at 9 | GP-EI label, direct-warm at 9 | GP-UCB label, direct-warm at 9 |', '|---|---:|---:|---:|---:|']
    for row in budgets:
        curves = {(x['method'], x['budget']): x for x in row['curves']}
        val = lambda method, budget: curves[(method, budget)]['parent_mean_loss']
        lines.append(f"| {row['run']} | {val('direct', 0):.6f} | {val('sobol_local/direct', 9):.6f} | {val('bayesian/direct', 9):.6f} | {val('finzgar_gp_ucb/direct', 9):.6f} |")
    lines += ['', 'The cold-start losses at nine queries are Sobol 0.609837, GP-EI-labeled 0.602952, and GP-UCB-labeled 0.618101. Bank warm starts yield 0.597805, 0.601833, and 0.614906, respectively—but source-global warm starts reproduce those values within 4.3×10⁻⁹. Thus these bank-warm gains do not isolate an ML advantage. The direct results vary with checkpoint; no pooled superiority over source-global is established.', '',
              '**Initialization coverage matters.** In these budget studies, Finžgar GP-UCB uses ten initial points and never reaches an adaptive query at maximum budget nine. The generic eight-dimensional GP-EI implementation also stays within its initial-design phase. Their algorithm labels must not be interpreted as evidence about Bayesian adaptation. The separate [25-query diagnostic follow-up](../paper_budget_validation_2026-09-18/SUMMARY.md) exercises both adaptive optimizers using the same checkpoints; it is not part of this frozen pilot.', '',
              'Each budget report records 1,296 online search calls plus 24 offline diagnostics, with no failed calls. Across six reports: 7,776 online calls and 144 diagnostics. Whole-study measured offline preparation is 76.964 seconds, shared across these reports and not six separate costs; it includes all acquisition/mechanism fits, not a minimal single-method training cost. No fully observed cold/warm pair attains a common requested threshold of 0.4, 0.5, or 0.6, so **all break-even estimates are unavailable**, not zero or infinite measured advantages.', '',
              '## Literature-oriented adapted baseline', '',
              'The separate literature campaign compares the Finžgar-inspired GP-UCB adaptation against uniform random on **four parents/eight records**, optimizer seeds 0/1/2, independent budgets four and eight: 96 optimizer runs, 576 completed objective calls, no failed queries. It is explicitly an embedded-task adaptation with reduced settings, not reproduction of the published numerical study. Here `n_initial=4`, so budget four uses only initialization while budget eight exercises four adaptive queries.', '',
              '| Budget | GP-UCB − random | 95% crossed parent×optimizer-seed CI |', '|---|---:|---|']
    for budget in literature['budgets']:
        c = budget['gp_ucb_minus_uniform_random']
        lines.append(f"| {budget['budget']} | {c['mean_difference']:+.6f} | [{c['ci_low']:+.6f}, {c['ci_high']:+.6f}] |")
    lines += ['', 'The four-query tie follows from shared initialization. The eight-query interval includes zero; only 200 bootstrap resamples were configured. No literature-baseline efficacy claim is supported by this pilot.', '',
              '## Reproduce and inspect', '', '```bash', 'python reports/paper_campaign_2026-09-18/reproduce.py', 'python reports/paper_campaign_2026-09-18/reproduce.py --check', '```', '',
              'The compact `results.json` includes all six budget curves and contrasts, source/transfer contrasts, input hashes, counts, and cost scopes. `pilot_evidence.json.gz` contains raw text reports, trajectories, receipts, manifests, and logs; `smoke_evidence.json.gz` is a separate integration run. The archive omits trained-weight binaries, dataset NPZ files, and figures. Hashes identify those omitted files but are not substitutes for them: table reconstruction works from the archive, while exact model replay requires regenerating or retrieving the binaries. No completed QPU, 16–20-qubit, or confirmatory efficacy result is claimed.', '']
    return {'results.json': json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n', 'RESULTS.md': '\n'.join(lines)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=HERE / 'pilot_evidence.json.gz')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    outputs = build(args.archive)
    if args.check:
        changed = [name for name, value in outputs.items() if not (HERE/name).exists() or (HERE/name).read_text() != value]
        if changed:
            raise SystemExit('Synopsis differs: ' + ', '.join(changed))
        print('Verified pilot synopsis against archived reports and hashes.')
    else:
        for name, value in outputs.items():
            (HERE/name).write_text(value)
        print('Wrote pilot RESULTS.md and results.json.')


if __name__ == '__main__':
    main()
