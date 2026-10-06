"""
One-off: rename the original score's 'norm_score' to 'final_score_norm' in saved results, so outputs written before
the rename keep working with the current code (no rerun needed). Changes, under result/:
    score files                header only (the files are large, so they are streamed)
    report_*/ tables           the score name inside them, and file names ending in '_norm_score'
Safe to run more than once. From the project root:
    python rename_norm_score.py
"""
import pathlib
import re
import shutil

OLD, NEW = 'norm_score', 'final_score_norm'

if __name__ == '__main__':
    res = pathlib.Path('result')
    n = 0
    for f in sorted(res.glob('*score*.csv')):
        with open(f) as fh:
            head = fh.readline()
            if not re.search(rf'(^|,){OLD}(,|$)', head.strip()):
                continue
            tmp = f.with_suffix('.tmp')
            with open(tmp, 'w') as out:
                out.write(re.sub(rf'(^|,){OLD}(?=,|$)', rf'\1{NEW}', head.rstrip('\n')) + '\n')
                shutil.copyfileobj(fh, out, 1 << 24)
        tmp.replace(f)
        n += 1
        print(f'header: {f}')
    for d in sorted(res.glob('report_*')):
        for f in sorted(d.glob('*.csv')):
            text = f.read_text()
            if re.search(rf'\b{OLD}\b', text):
                f.write_text(re.sub(rf'\b{OLD}\b', NEW, text))
                n += 1
            if f.stem.endswith(f'_{OLD}'):
                f.rename(f.with_name(f.stem[:-len(OLD)] + f'{NEW}.csv'))
                n += 1
    print(f'{n} changes')
