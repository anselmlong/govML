import sys
sys.path.insert(0, '/home/ubuntu/govml')
from batch_ml import run_one

for rid in ['d_9e7de44094f876f6804b8b5bcee45c81']:
    try:
        r = run_one(rid, rid, no_correlation=True)
        if not r:
            print(rid, '-> None (skipped)'); continue
        m = r['metrics']
        print('###', rid, '| target:', r['target'], '| task:', r['task_type'], '| best:', r['best_model'])
        print('    verdict:', r['verdict_tone'], '| R2:', round(m.get('R2', float('nan')), 3),
              '| MAE:', round(m.get('MAE', 0), 2))
    except Exception as e:
        import traceback; print(rid, '-> EXC', repr(e)[:120]); traceback.print_exc(limit=2)