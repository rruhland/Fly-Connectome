#include <algorithm>
#include <cmath>
#include <cstdint>

// Integrate the stock NumPy approximate trajectories one at a time. Emit
// only useful rows, leaving the stock weighted NumPy/BLAS distance to Python.
extern "C" int approximate_flight(const double* pos, const double* vel,
                                  const double* spin, const double* goal, int count,
                                  double drag, double lift, double radius, double top,
                                  double* records, int* index, double* cross, double* land) noexcept {
    constexpr double dt=.010;
    const double depth_min=std::min(-1.30,goal[1]-.08);
    int written=0;
    for (int i=0;i<count;++i) {
        double p[3],v[3],w[3];
        std::copy_n(pos+3*i,3,p);
        std::copy_n(vel+3*i,3,v);
        std::copy_n(spin+3*i,3,w);
        int bounces=0;
        for (int step=0;step<105;++step) {
            const double coeff=-drag*std::sqrt((v[0]*v[0]+v[1]*v[1])+v[2]*v[2]);
            double a[3]={coeff*v[0]+lift*(w[1]*v[2]-w[2]*v[1]),
                         coeff*v[1]+lift*(w[2]*v[0]-w[0]*v[2]),
                         coeff*v[2]+lift*(w[0]*v[1]-w[1]*v[0])};
            a[2]-=9.81;
            double q[3];
            for (int j=0;j<3;++j) {
                q[j]=p[j]+dt*(v[j]+.5*dt*a[j]);
                v[j]+=dt*a[j];
            }
            if (p[1]>=0.&&q[1]<0.) {
                const double u=p[1]/(p[1]-q[1]);
                cross[i]=p[2]+u*(q[2]-p[2]);
            }
            if (p[2]>=top&&q[2]<top&&std::abs(q[0])<.765&&std::abs(q[1])<1.37) {
                if (++bounces==1) {
                    land[i]=q[1];
                    const double sx=v[0]-radius*(w[1]*1.-w[2]*0.);
                    const double sy=v[1]-radius*(w[2]*0.-w[0]*1.);
                    const double n=std::sqrt(sx*sx+sy*sy);
                    const double impulse=std::min(.24,((.18*1.86)*-v[2])/std::max(n,.01));
                    v[0]-=impulse*sx;
                    v[1]-=impulse*sy;
                    const double dx=-impulse*sx,dy=-impulse*sy;
                    w[0]+=(1./(.4*radius))*(0.*0.-(-1.)*dy);
                    w[1]+=(1./(.4*radius))*((-1.)*dx-0.*0.);
                    w[2]+=(1./(.4*radius))*(0.*dy-0.*dx);
                    q[2]=2*top-q[2];
                    v[2]*=-.86;
                }
            }
            for (int j=0;j<3;++j) {
                p[j]=q[j];
                w[j]*=1.-.18*dt;
            }
            if (bounces==1 && p[1]<-.65 && p[1]>depth_min && p[2]>top+.045&&p[2]<top+.42) {
                double* at=records+5*written;
                at[0]=std::abs(p[0]-goal[0]); at[1]=std::abs(p[1]-goal[1]); at[2]=std::abs(p[2]-goal[2]);
                at[3]=.60*std::abs(v[1]+goal[3]);
                at[4]=.002*std::max(0.,40.-w[0]);
                index[written++]=i;
            }
        }
    }
    return written;
}
