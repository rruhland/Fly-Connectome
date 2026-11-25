"""Recover final metadata after the completed course's hash type error; no steps."""
from pathlib import Path
from .tt51_low_reverse_curriculum import arm,extract,plain
from .tt29_rally_physics import BladeIK


def run():
    path=Path('artifacts/tt51_low_reverse/course.json.gz');d=arm._read(path)
    if len(d['rows'])!=18 or d.get('in_progress'):raise ValueError('Incomplete course')
    source=path.parent/'executed_source.py'
    if arm._hash(source)!=d['source_sha256']:raise ValueError('Missing exact executed source')
    templates=list(d['initial_templates'])
    with arm.mpc.centered(arm._read(arm.ARCHIVE)['ready']):
        a=arm.mpc.arena(detail='cad')
        try:
            ik=BladeIK(a)
            for r in d['rows']:
                v=r['variants'].get('growing',{})
                if v.get('new_rgb_template_added'):
                    templates.append(plain(extract(v['observations'],ik,
                        f"own-reward-{r['stage']}-{r['index']}",r['cut'])))
        finally:a.close()
    d.update(complete=True,status='completed_reverse_curriculum_diagnostic_no_official_promotion',
        final_templates=templates,source_frozen=arm._hash(source)==d['source_sha256'],
        dependencies_frozen=all(arm._hash(Path(p))==h for p,h in d['dependency_hashes'].items()),
        final_metadata_recovery=dict(source_sha256=arm._hash(Path(__file__)),
            reason='All18 cases persisted; original finish failed hashing string path. No physical reruns. Final times are last persisted completed-row lower bounds; template aggregation reconstructed from retained actual rewarded observations.',
            new_native_steps=0))
    arm._write(path,d)
    print('recovered',len(templates),'templates',d['actual_control_ticks'],'paid ticks',flush=True)


if __name__=='__main__':run()
