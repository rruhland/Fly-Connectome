// Contact-free, intact stock arm. Real controller substeps and actuator envelope.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace {
constexpr int N=6;
struct Motor {
    mjModel* m;
    const int* poses;
    const int* dofs;
    const double* step;
    const double* low;
    const double* high;
    double full_speed;
    int micro, racket;
    double force[2*N];
    Motor(mjModel* model, const int* p, const int* v, const double* s, const double* l,
          const double* h, double speed, int k, int g)
        : m(model), poses(p), dofs(v), step(s), low(l), high(h), full_speed(speed), micro(k), racket(g) {
        std::copy_n(m->actuator_forcerange, 2*N, force);
    }
    void set_force(mjData* d) {
        for(int i=0; i<N; ++i) {
            const double speed=d->qvel[dofs[i]]/full_speed;
            m->actuator_forcerange[2*i]=force[2*i]*std::clamp(1.+speed,0.,1.);
            m->actuator_forcerange[2*i+1]=force[2*i+1]*std::clamp(1.-speed,0.,1.);
        }
    }
    void tick(mjData* d, const double* rates) {
        for(int j=0; j<micro; ++j) {
            for(int i=0; i<N-1; ++i)
                d->ctrl[i]=std::clamp(d->ctrl[i]+step[i]*std::clamp(rates[i],-.8,.8),low[i],high[i]);
            set_force(d);
            mj_step(m,d);
        }
    }
    void state(mjData* d, float* x) {
        for(int i=0; i<N-1; ++i) {
            x[i]=d->qpos[poses[i]];
            x[i+5]=d->qvel[dofs[i]];
            x[i+10]=d->ctrl[i];
        }
    }
};
struct Random {
    uint64_t state;
    double next() {
        state+=0x9e3779b97f4a7c15ULL;
        uint64_t x=state;
        x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
        x=(x^(x>>27))*0x94d049bb133111ebULL;
        return ((x^(x>>31))>>11)*0x1.0p-53;
    }
};
}

extern "C" void tt88_babble(const mjModel* nominal, const mjData* initial,
      const int* poses, const int* dofs, const double* step, const double* low, const double* high,
      double speed,int micro,int racket, int episodes,int ticks, uint64_t seed,float* output) {
    mjModel* model=mj_copyModel(nullptr,nominal);
    mjData* data=mj_makeData(model);
    Motor body(model,poses,dofs,step,low,high,speed,micro,racket);
    Random random{seed};
    for(int episode=0; episode<episodes; ++episode) {
        mj_copyData(data,model,initial);
        std::array<double,5> u{}, prev{}, goal{}, directions{};
        int stop=-1, mode=0;
        double amplitude=0.;
        for(int t=0; t<ticks; ++t) {
            if(t>=stop) {
                mode=std::min(7,int(random.next()*8));
                amplitude=.28+.72*random.next();
                stop=t+5+int(42*random.next());
                for(int j=0; j<5; ++j) {
                    // Alternate broad workspace references, nearby excursions and
                    // coherent reversing accelerations around existing physical states.
                    double target=low[j]+(high[j]-low[j])*random.next();
                    if(mode==1 || mode==6) target=initial->ctrl[j]+.55*(target-initial->ctrl[j]);
                    if(mode==4 || mode==7) target=data->ctrl[j]+.80*(2.*random.next()-1.);
                    goal[j]=std::clamp(target,low[j],high[j]);
                    directions[j]=2.*random.next()-1.;
                }
            }
            for(int j=0; j<5; ++j) {
                const double toward=std::clamp((goal[j]-data->ctrl[j])/(10.*body.micro*body.step[j]),-.8,.8);
                const double pulse=2.*random.next()-1.;
                if(mode<=1) u[j]=toward*(.70+.30*random.next())+.30*pulse;
                else if(mode<=3) u[j]=.70*prev[j]+amplitude*(.26*pulse+.15*toward+.12*directions[j]);
                else if(mode<=5) u[j]=toward+.45*amplitude*pulse;
                else u[j]=amplitude*(.58*directions[j]+.47*pulse)+.29*toward;
                u[j]=std::clamp(u[j],-.8,.8);
            }
            float* row=output+(static_cast<size_t>(episode)*ticks+t)*35;
            body.state(data,row);
            for(int j=0; j<5; ++j) row[15+j]=u[j];
            body.tick(data,u.data());
            prev=u;
            body.state(data,row+20);
        }
    }
    mj_deleteData(data);
    mj_deleteModel(model);
}

extern "C" void tt88_evaluate(const mjModel* nominal,const mjData* initial,
      const int* poses,const int* dofs,const double* step,const double* low,const double* high,
      double speed,int micro,int racket,const double* commands,int trials,int ticks,double* out) {
    mjModel* model=mj_copyModel(nullptr,nominal);
    mjData* data=mj_makeData(model);
    Motor body(model,poses,dofs,step,low,high,speed,micro,racket);
    for(int i=0; i<trials; ++i) {
        mj_copyData(data,model,initial);
        for(int t=0; t<ticks; ++t) body.tick(data,commands+(static_cast<size_t>(i)*ticks+t)*5);
        mj_forward(model,data);
        double* x=out+22*i;
        for(int j=0; j<5; ++j) {
            x[j]=data->qpos[poses[j]];
            x[j+5]=data->qvel[dofs[j]];
        }
        const double* p=data->geom_xpos+3*racket;
        const double* R=data->geom_xmat+9*racket;
        std::copy_n(p,3,x+10);
        for(int j=0;j<3;++j) x[13+j]=R[3*j+2];
        mj_objectVelocity(model,data,mjOBJ_GEOM,racket,x+16,0);
    }
    mj_deleteData(data);
    mj_deleteModel(model);
}
