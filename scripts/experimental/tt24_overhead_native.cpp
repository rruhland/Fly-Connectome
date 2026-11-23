// A ceiling-facing two-racket exchange. The original donor evaluator is retained for controls.
#define tt24_donor_evaluate tt24_plain_evaluate
#include "tt24_donor_native.cpp"
#undef tt24_donor_evaluate

namespace {
void exchange(Donor& world, const double* weights, const double* initial, const double* physics,
              int cap, double* out, double* tape=nullptr) {
    mjModel* m=world.model;
    mjData* d=world.data;
    mj_resetData(m,d);
    m->geom_solref[world.blade*mjNREF+1]=-physics[1];
    mj_setState(m,d,initial,mjSTATE_INTEGRATION);
    mj_forward(m,d);
    int opponent=mj_name2id(m,mjOBJ_GEOM,"opponent_racket");
    int mocap=m->body_mocapid[m->geom_bodyid[opponent]];
    double* hand=d->mocap_pos+3*mocap;
    bool precise=m->geom_size[3*opponent]<.06;
    double speed_limit=precise?.12:.50;
    const double* pad=d->geom_xpos+3*world.blade;
    double home[]={pad[0],pad[1],pad[2]};
    const double ceiling=hand[2];
    double* ball=d->qpos+world.position;
    double* velocity=d->qvel+world.velocity;
    double issued[5]={}, action[5], state[11];
    int last=0, receives=0, pairs=0, tick=0;
    bool touching=false, done=false;
    for (; tick<cap && !done; ++tick) {
        mj_kinematics(m,d);
        for(int i=0;i<3;++i) {
            state[i]=ball[i]-home[i];
            state[i+3]=velocity[i];
            state[i+6]=pad[i]-home[i];
        }
        const double* normal=d->geom_xmat+9*world.blade;
        double sign=normal[8]>=0.?1.:-1.;
        state[9]=sign*normal[2]; state[10]=sign*normal[5];
        command(state,weights,issued,action);
        if(tape) std::copy(action,action+5,tape+5*tick);
        // The partner tracks laterally at finite speed and moves its face vertically.
        // All momentum exchange takes place through actual MuJoCo contacts.
        for(int sub=0;sub<kSubsteps && !done;++sub) {
            for(int i=0;i<5;++i)
                d->ctrl[i]=std::clamp(d->ctrl[i]+action[i]*kRate*m->opt.timestep,
                                      m->actuator_ctrlrange[2*i],m->actuator_ctrlrange[2*i+1]);
            for(int i=0;i<2;++i) {
                const double aim=precise?std::clamp(ball[i],home[i]-.025,home[i]+.025):ball[i];
                hand[i]+=std::clamp(aim-hand[i],-speed_limit*m->opt.timestep,speed_limit*m->opt.timestep);
            }
            hand[2]=ceiling+.008*std::sin(17.*d->time);
            double factor=physics[0]*std::sqrt(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2]);
            for(int i=0;i<3;++i) d->xfrc_applied[6*world.body+i]=factor*velocity[i];
            double incoming=velocity[2];
            mj_step(m,d);
            int hit=0;
            for(int j=0;j<d->ncon;++j) {
                const mjContact& c=d->contact[j];
                if(c.efc_address<0) continue;
                int other=c.geom[0]==world.ball?c.geom[1]:(c.geom[1]==world.ball?c.geom[0]:-1);
                if(other==-1) continue;
                if(other==world.blade) hit=1;
                else if(other==opponent) hit=2;
                else done=true;
            }
            if(hit && !touching) {
                if(hit==last || (hit==1 && incoming>-.20) || (hit==2 && (incoming<.20 || last!=1))) done=true;
                else if(hit==1) ++receives;
                else ++pairs;
                last=hit;
            }
            touching=hit!=0;
            if(ball[2]<.51 || ball[2]>2.1 || std::abs(ball[0])>.66 || ball[1]<=-1.65 || ball[1]>=-.33)
                done=true;
        }
    }
    out[0]=pairs; out[1]=pairs; out[2]=tick; out[3]=receives;
}
}
extern "C" void tt24_donor_evaluate(void* handle,const double* weights,int population,const double* states,
                                    const double* settings,int starts,int cap,double* outputs) {
    auto* donor=static_cast<Donor*>(handle);
    for(int i=0;i<population;++i)
        for(int s=0;s<starts;++s)
            exchange(*donor,weights+i*kJoints*kFeatures,states+s*donor->state_size,settings+2*s,cap,
                     outputs+4*(i*starts+s));
}

extern "C" void tt24_overhead_trace(void* handle,const double* weights,const double* initial,
                                    const double* settings,int cap,double* outputs,double* tape) {
    exchange(*static_cast<Donor*>(handle),weights,initial,settings,cap,outputs,tape);
}
