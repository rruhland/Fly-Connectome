// One isolated encoder-tethered body, no ball contacts and no opponent feedback.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

extern "C" void body_terminals(const mjModel* nominal, const mjData* start, const double* goals,
    int candidates, int frames, int micro, int racket, const int* motors, const int* dofs,
    const double* increments, const double* low, const double* high, const double* nominal_limits,
    double no_load, double slew, double* output) {
    constexpr int width=9, axes=6;
    mjModel* m = mj_copyModel(nullptr, nominal);
    mjData* d = mj_makeData(m);
    for(int trial=0; trial<candidates; ++trial) {
        mj_copyData(d, m, start);
        std::array<double, axes> rate;
        bool valid=true;
        for(int t=0; t<frames && valid; ++t) {
            for(int j=0; j<axes; ++j) {
                const int a=motors[j];
                const double target=std::clamp(goals[(trial*frames+t)*axes+j], low[a], high[a]);
                rate[j]=increments[a]*std::clamp((target-d->ctrl[a])/(micro*increments[a]),-slew,slew);
            }
            for(int sub=0; sub<micro && valid; ++sub) {
                for(int j=0; j<axes; ++j) {
                    const int a=motors[j];
                    d->ctrl[a]=std::clamp(d->ctrl[a]+rate[j],low[a],high[a]);
                    const double spin=d->qvel[dofs[j]]/no_load;
                    m->actuator_forcerange[2*a]=nominal_limits[2*j]*std::clamp(1.+spin,0.,1.);
                    m->actuator_forcerange[2*a+1]=nominal_limits[2*j+1]*std::clamp(1.-spin,0.,1.);
                }
                mj_step(m,d);
                for(int j=0; j<axes; ++j)
                    valid=valid && std::isfinite(d->qvel[dofs[j]]) && std::abs(d->qvel[dofs[j]]) <= no_load+2.;
            }
        }
        double* out=output+trial*width;
        if(!valid) {
            std::fill(out,out+width,std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        mj_forward(m,d);
        const double* p=d->geom_xpos+3*racket;
        const double* n=d->geom_xmat+9*racket;
        const double sign=n[5]<0. ? -1. : 1.;
        double twist[6];
        mj_objectVelocity(m,d,mjOBJ_GEOM,racket,twist,0);
        for(int j=0; j<3; ++j) {
            out[j]=p[j]; out[3+j]=sign*n[2+3*j]; out[6+j]=twist[3+j];
        }
    }
    mj_deleteData(d);
    mj_deleteModel(m);
}
