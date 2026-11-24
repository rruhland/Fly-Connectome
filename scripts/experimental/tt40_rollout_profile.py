"""Add opt-in timing/counter probes to a throwaway native rollout library."""

import argparse
from pathlib import Path


def instrument(source):
    text = source
    edits = (
        ('#include <vector>', '#include <vector>\n#include <chrono>\n#include <cstdio>'),
        ('    const int ball = ids[0]', '''    const auto clock = []() { return std::chrono::steady_clock::now(); };
    const auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b-a).count(); };
    const int ball = ids[0]'''),
        ('        // Each worker', '''        const auto begin=clock();
        double copies=0., before=0., after=0., guard=0., narrow=0.;
        long pre_n=0,post_n=0,guards=0,calls=0;
        // Each worker'''),
        ('        #pragma omp for', '''        const double allocate=seconds(begin,clock());
        #pragma omp for'''),
        ('            mj_copyData(d, model, source);', '''            const auto copy_start=clock();
            mj_copyData(d, model, source);
            copies+=seconds(copy_start,clock());'''),
        ('                    mj_step(model, d);', '''                    const auto step_start=clock();
                    mj_step(model, d);
                    if (struck) { after+=seconds(step_start,clock()); ++post_n; }
                    else { before+=seconds(step_start,clock()); ++pre_n; }'''),
        ('                        const double threshold', '''                        const auto guard_start=clock();
                        ++guards;
                        const double threshold'''),
        ('                            if (mj_geomDistance(model, d, first_geom, second_geom, horizon, nullptr) < threshold) {',
         '''                            const auto distance_start=clock();
                            const double dist=mj_geomDistance(model, d, first_geom, second_geom, horizon, nullptr);
                            narrow+=seconds(distance_start,clock()); ++calls;
                            if (dist < threshold) {'''),
        ('                    }\n                    if (fault) break;\n                    const double x',
         '''                        guard+=seconds(guard_start,clock());
                    }
                    if (fault) break;
                    const double x'''),
        ('        if (local) mj_deleteModel(local);', '''        if (local) mj_deleteModel(local);
        std::fprintf(stderr, "profile count=%d total=%.7f alloc=%.7f copies=%.7f step_in=%.7f step_out=%.7f guard=%.7f narrow=%.7f steps_in=%ld steps_out=%ld checks=%ld narrow_calls=%ld\\n",
                     count,seconds(begin,clock()),allocate,copies,before,after,guard,narrow,pre_n,post_n,guards,calls);'''),
    )
    for old,new in edits:
        if text.count(old)!=1:
            raise ValueError(f'Timing probe no longer matches exactly once: {old}')
        text=text.replace(old,new,1)
    return text


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    a.output.write_text(instrument(a.source.read_text()))
