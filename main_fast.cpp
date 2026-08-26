#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <thread>
#include <iomanip>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <Accelerate/Accelerate.h>
using namespace std;
#define all(v) v.begin(), v.end()
using ll=long long;
int INF=0x3f3f3f3f;
float SENTINEL=-MAXFLOAT;
thread_local mt19937 rng(hash<thread::id>{}(this_thread::get_id())^random_device{}());
float gaussian_noise(float mean, float stddev){
    normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}
vector<float> actuator={
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};
thread_local uniform_real_distribution<float> disf(0.0f, 1.0f);
float dt=0.01f;
void matvec(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int idx1, int idx2){
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*m), 1, 0.0f, c.data(), 1);
}
void matvec_transpose(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int idx1, int idx2){
    cblas_sgemv(CblasRowMajor, CblasTrans, n, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*n), 1, 0.0f, c.data(), 1);
}
void matmat(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int p, int idx1, int idx2){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, p, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*m*p), p, 0.0f, c.data(), p);
}
void matmat_transpose(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int p, int idx1, int idx2){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, p, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*p*m), m, 0.0f, c.data(), p);
}
float mag(const vector<float>&a, int start_idx, int size){
    return cblas_sdot(size, a.data()+start_idx, 1, a.data()+start_idx, 1);
}
void delta_rule(vector<float> &a, const vector<float>&x, const vector<float>&err, float lr, int n, int m, int idx){
    //a: nxm, x: mx1, err: nx1
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            a[i*m+j]+=-lr*dt*err[i]*x[idx*m+j];
        }
    }
}
void delta_rule_tanh(vector<float> &a, const vector<float>&x, const vector<float>&err, const vector<float>& y, float lr, float s, int n, int m, int idx1, int idx2){
    //a: nxm, x: mx1, err: nx1
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            a[i*m+j]+=-lr*dt*err[i]*(1-(y[idx2*n+i]*y[idx2*n+i])/(s*s))*x[idx1*m+j];
        }
    }
}
vector<float> randvec(int l, float mg){
    vector<float> ret(l, 0.0f); for (int i=0;i<l;i++) ret[i]=gaussian_noise(0, mg);
    return ret;
}
vector<float> randeye(int d, float mg){
    vector<float> ret(d*d,0.0f); for (int i=0;i<d;i++) ret[i*d+i]=1.0f;
    for (int i=0;i<d*d;i++) ret[i]+=gaussian_noise(0.0f,mg);
    return ret;
}
vector<float> randvec_u(int l, int mg){
    uniform_real_distribution<float> disi(-mg, mg);
    vector<float> ret(l, 0.0f); for (int i=0;i<l;i++) ret[i]=disi(rng);
    return ret;
}
struct edge{
    vector<float> w;
    vector<float> u; 
    vector<float> v;
};
class lupus_k{
    public:
        int n, d;
        vector<float> chain; //nxdxd
        vector<float> res; //(n+1)xdx1
        float tanh_mag;
        vector<float> err;
        vector<float> ograd;
        float lr;
        float noise_mag;
        void reset(){
            chain.assign(n*d*d,0.0f);
            for (int i=0;i<n;i++) for (int j=0;j<d;j++) chain[i*d*d+j*d+j]=1.0f;
            for (int i=0;i<n*d*d;i++) chain[i]+=gaussian_noise(0.0f, noise_mag);
            res.assign((n+1)*d,0.0f);
            err.assign(d,0.0f);
            ograd.assign(d,0.0f);
        }
        lupus_k(int un, int ud, float ulr, float tm, float nm){
            n=un; d=ud; lr=ulr; tanh_mag=tm; noise_mag=nm;
            reset();
        }
        vector<float> forward(vector<float> h, vector<float> o){
            for (int i=0;i<n;i++){
                for (int j=0;j<d;j++) res[i*d+j]=h[j];
                vector<float> nh(d, 0.0f);
                matvec(chain, h, nh, d, d, i, 0);
                for (int j=0;j<d;j++) nh[j]=tanh_mag*tanhf(nh[j]/tanh_mag);
                swap(h, nh);
            }
            for (int j=0;j<d;j++) res[n*d+j]=h[j];
            for (int i=0;i<d;i++){
                err[i]=h[i]-o[i];
            }
            return h;
        }
        void backward_learn(){
            vector<float> grad=err;
            for (int i=n-1;i>=0;i--){
                for (int j=0;j<d;j++) grad[j]*=1.0f-(res[(i+1)*d+j]*res[(i+1)*d+j])/(tanh_mag*tanh_mag);
                ograd=grad;
                vector<float> ngrad(d, 0.0f);
                matvec_transpose(chain, grad, ngrad, d, d, i, 0);
                for (int j=0;j<d;j++){
                    for (int k=0;k<d;k++){
                        chain[i*d*d+j*d+k]-=lr*ograd[j]*res[i*d+k];
                    }
                }
                swap(grad, ngrad);
            }
        }
};
class lupus_w{
    public:
        int n, d;
        vector<unordered_map<int, edge>> adj;
        vector<float> h; //nxdx1
        vector<float> force; //nxdx1
        vector<float> received_signal; //dx1
        vector<float> chl_err; //dx1
        vector<float> scaled_chl_err_p; //dx1
        vector<float> par_change; //dx1
        vector<float> precision; //dx1
        vector<float> chl_err_p; //dx1
        vector<bool> fixed; //nx1
        vector<float> prev_env; //dx1
        vector<float> prev_act; //dx1
        vector<int> deg;
        map<pair<int, int>, pair<int, int>> mirror;
        int tick=0;
        lupus_k k=lupus_k(0, 0, 0.0f, 0.0f, 0.0f);
        float slow_learn, fast_learn, eps, tanh_mag, noise_mag, min_noise, max_noise;
        bool has_prev=false;
        void add_mirror_edge(int x1, int y1, int x2, int y2){
            adj[x1][y1]=edge{randeye(d,0.05f/sqrtf(d)),vector<float>(d,0.0f),vector<float>(d,0.0f)};
            adj[x2][y2]=adj[x1][y1];
            mirror[{x1, y1}]={x2, y2};
            mirror[{x2, y2}]={x1, y1};
        }
        void add_edge(int x1, int y1){
            adj[x1][y1]=edge{randeye(d,0.05f/sqrtf(d)),vector<float>(d,0.0f),vector<float>(d,0.0f)};
        }
        void reset(){
            tick=0;
            adj.assign(n, unordered_map<int, edge>{});
            add_mirror_edge(1,3,1,6);
            add_mirror_edge(3,4,6,7);
            add_mirror_edge(4,5,7,8);
            add_mirror_edge(5,0,8,2);
            h.assign(n*d, 0.0f); // h=randvec(n*d, 1.0f/sqrtf(d));
            force.assign(n*d, 0.0f);
            received_signal.assign(d, 0.0f);
            chl_err.assign(d, 0.0f);
            scaled_chl_err_p.assign(d, 0.0f);
            par_change.assign(d, 0.0f);
            precision.assign(d, 0.0f);
            chl_err_p.assign(d, 0.0f);
            prev_env.assign(d, 0.0f);
            prev_act.assign(d, 0.0f);
            fixed.assign(n, false); fixed[0]=true; fixed[2]=true;
            k.reset();
            deg.assign(n,0);
            for (int i=0;i<n;i++){
                for (auto[j,_]:adj[i]){
                    deg[i]++; deg[j]++;
                }
            }
            has_prev=false;
        }
        lupus_w(float un, float ud, float sl, float fl, float e, float tm, float nm, float min_n, float max_n, lupus_k uk){
            n=un; d=ud;
            slow_learn=sl; fast_learn=fl; eps=e; tanh_mag=tm; k=uk; noise_mag=nm; min_noise=min_n; max_noise=max_n;
            reset();
        }
        void forward(){
            fill(all(force), 0.0f);
            for (int par=0;par<n;par++){
                for (auto& [i, e]:adj[par]){
                    auto& [w, u, v]=e;
                    for (int j=0;j<d;j++) precision[j]=1.0f/(max(0.0f, u[j]-v[j]*v[j])+eps);
                    // for (int j=0;j<d;j++) precision[j]=1.0f;
                    matvec(w, h, received_signal, d, d, 0, par);
                    for (int j=0;j<d;j++) received_signal[j]=tanh_mag*tanhf(received_signal[j]/tanh_mag);
                    for (int j=0;j<d;j++) {
                        chl_err[j]=received_signal[j]-h[i*d+j];
                        chl_err_p[j]=chl_err[j]*precision[j];
                        force[i*d+j]+=chl_err_p[j];
                        u[j]+=dt*(chl_err[j]*chl_err[j]-u[j]);
                        v[j]+=dt*(chl_err[j]-v[j]);
                        scaled_chl_err_p[j]=chl_err_p[j]*(1-(received_signal[j]*received_signal[j])/(tanh_mag*tanh_mag));
                    }
                    matvec_transpose(w, scaled_chl_err_p, par_change, d, d, 0, 0);
                    for (int j=0;j<d;j++) force[par*d+j]-=par_change[j];
                    delta_rule_tanh(w, h, chl_err_p, received_signal, slow_learn, tanh_mag, d, d, par, 0);
                    if (mirror.count({par, i})){
                        auto [jj, kk]=mirror[{par, i}];
                        adj[jj][kk].w=w;
                    }
                }
            }
            for (int i=0;i<n;i++){
                for (int j=0;j<d;j++) force[i*d+j]/=max(1, deg[i]);
                if (!fixed[i]) for (int j=0;j<d;j++) h[i*d+j]+=dt*fast_learn*force[i*d+j];
            }
        }
        vector<float> generate(vector<float> ipt, vector<float> prior, int settle_steps){
            if (has_prev){
                vector<float> env_change(d,0.0f);
                for (int i=0;i<d;i++) env_change[i]=(ipt[i]-prev_env[i])/dt;
                k.forward(env_change, prev_act);
                k.backward_learn();
            }
            for (int i=0;i<d;i++) h[0*d+i]=ipt[i];
            for (int i=0;i<d;i++) h[2*d+i]=prior[i];
            for (int i=0;i<settle_steps;i++) forward();
            vector<float> ret(d,0.0f);
            vector<float> sense_force(d,0.0f); for (int i=0;i<d;i++) sense_force[i]=force[0*d+i];
            vector<float> act_k=k.forward(sense_force, sense_force);
            float act_rms=sqrtf(mag(act_k, 0, d)/d);
            for (int i=0;i<d;i++) {
                act_k[i]+=gaussian_noise(0.0f,min(max_noise, max(min_noise*expf(-tick/100000.0f), noise_mag*act_rms)));
                ret[i]=dt*act_k[i];
            }
            prev_act=act_k;
            prev_env=ipt;
            has_prev=true;
            tick++;
            return ret;
        }
};
vector<array<float,4>> gettrial(int len, unsigned int seed){
    mt19937 target_rng(seed);
    uniform_real_distribution<float> dist_target(-1.0f, 1.0f);
    vector<array<float,4>> ret{};
    while (ret.size()<len){
        ret.push_back({dist_target(target_rng), dist_target(target_rng), dist_target(target_rng), dist_target(target_rng)});
    }
    return ret;
}
vector<array<float,4>> trial=gettrial(30, 1225);
//vector<float> angles={0.0f,0.25f,0.5f,0.5277778f,0.5555556f,0.5833333f,0.6111111f,0.625f,0.6388889f,0.6527778f,0.6666667f,0.6805556f,0.6944444f,0.7083333f,0.7222222f,0.7361111f,0.75f};
int total=100; int succeeded=0;
vector<int> endsat(trial.size()+1,0);
vector<int> eachend(total, 0);
vector<lupus_w> sexti{};
float l1=1.0f, l2=1.0f;
float pival=3.141592653589793;
bool keketrial=false;
int main(){
    if (keketrial){
        lupus_k keke(3, 4, 0.03f, 8.0f, 0.01f);
        for (int i=0;i<5000;i++){
            vector<float> ipt=randvec_u(4, 1.0f);
            vector<float> opt(4,0.0f); matvec(actuator, ipt, opt, 4, 4, 0, 0);
            keke.forward(opt, ipt); keke.backward_learn();
        }
        cout<<"forward generation:\n";
        for (int i=0;i<20;i++){
            vector<float> ipt=randvec_u(4, 1.0f);
            vector<float> opt(4,0.0f); matvec(actuator, ipt, opt, 4, 4, 0, 0);
            vector<float> gen=keke.forward(opt, ipt);
            cout<<"generated: \n";
            for (int j=0;j<4;j++) cout<<gen[j]<<' ';
            cout<<'\n';
            cout<<"expected: \n";
            for (int j=0;j<4;j++) cout<<ipt[j]<<' ';
            cout<<'\n';
        }
        // cout<<"0 -> 2\n";
        // for (int i=0;i<4;i++){
        //     for (int j=0;j<4;j++){
        //         cout<<keke.adj[0][2].w[i*4+j]<<' ';
        //     }
        //     cout<<'\n';
        // }
        // cout<<"2 -> 1\n";
        // for (int i=0;i<4;i++){
        //     for (int j=0;j<4;j++){
        //         cout<<keke.adj[2][1].w[i*4+j]<<' ';
        //     }
        //     cout<<'\n';
        // }
    } else{
        for (int _=0;_<total;_++){
            lupus_k keke(3, 4, 0.03f, 8.0f, 0.01f);
            lupus_w sextus(9, 4, 0.01f, 5.0f, 1.0f, 8.0f, 0.1f, 0.05f, 0.2f, keke);
            // float q1=-0.6f, q2=1.2f;
            // float cx=l1*cosf(q1)+l2*cosf(q1+q2);
            // float cy=l1*sinf(q1)+l2*sinf(q1+q2);
            float cx=0.0f, cy=0.0f;
            float cw=0.0f, cz=0.0f;
            float vw=0.0f, vx=0.0f, vy=0.0f, vz=0.0f;
            bool done=true;
            for (int i=0;i<trial.size();i++){
                auto [goalw, goalx, goaly, goalz]=trial[i];
                int timer=0;
                bool converged=false;
                for (int j=0;j<100000;j++){
                    // float theta=pival*1.0f*tanhf(cx);
                    //float theta=angles[i/10]*pival;
                    vector<float> sense(sextus.d,0.0f);
                    // sense[0]=q1; sense[1]=q2; sense[2]=cx; sense[3]=cy;
                    sense[0]=cw; sense[1]=cx; sense[2]=cy; sense[3]=cz;
                    // sense[2]=cosf(theta); sense[3]=sinf(theta);
                    vector<float> want(sextus.d,0.0f);
                    // want[0]=q1; want[1]=q2; want[2]=goalx; want[3]=goaly;
                    want[0]=goalw; want[1]=goalx; want[2]=goaly; want[3]=goalz;
                    vector<float> mv=sextus.generate(sense, want, 1);
                    // if (j%500==0) cout<<q1<<' '<<q2<<' '<<goalx<<' '<<goaly<<' '<<cx<<' '<<cy<<' '<<mv[0]<<' '<<mv[1]<<'\n';
                    // q1+=mv[0]; q2+=mv[1];
                    // q1=remainderf(q1, 2.0f*pival);
                    // q2=remainderf(q2, 2.0f*pival);
                    // cx=l1*cosf(q1)+l2*cosf(q1+q2);
                    // cy=l1*sinf(q1)+l2*sinf(q1+q2);
                    // cx+=mv[0]*cosf(theta)-mv[1]*sinf(theta);
                    // cy+=mv[0]*sinf(theta)+mv[1]*cosf(theta);
                    // cx-=mv[0]; cy-=mv[1];
                    vector<float> mv_j(sextus.d,0.0f);
                    matvec(actuator, mv, mv_j, sextus.d, sextus.d, 0, 0);
                    cw+=mv_j[0]; cx+=mv_j[1]; cy+=mv_j[2]; cz+=mv_j[3];
                    // vw+=mv_j[0]; vx+=mv_j[1]; vy+=mv_j[2]; vz+=mv_j[3];
                    // cw+=vw*dt; cx+=vx*dt; cy+=vy*dt, cz+=vz*dt;
                    if (abs(cw-goalw)<0.01f && abs(cx-goalx)<0.01f && abs(cy-goaly)<0.01f && abs(cz-goalz)<0.01f) timer++;
                    // if (abs(cx-goalx)<0.01f && abs(cy-goaly)<0.01f) timer++;
                    else timer=0;
                    if (timer>200) {
                        converged=true; break;
                    }
                }
                if (!converged){
                    done=false;
                    endsat[i]++;
                    eachend[_]=i;
                    break;
                }
            }
            if (done) {
                endsat[trial.size()]++;
                eachend[_]=trial.size();
                succeeded++;
                sexti.push_back(sextus);
            }
        }
        for (int i=0;i<trial.size();i++) {
            cout<<'('<<trial[i][0]<<','<<trial[i][1]<<','<<trial[i][2]<<','<<trial[i][3]<<"): "<<endsat[i]<<'\n';
        }
        cout<<succeeded<<'/'<<total<<'\n'<<setprecision(4)<<100.0f*succeeded/total<<"%\n";
    }
    return 0;
}
/*
clang++ -std=c++23 -O3 -Wall -DACCELERATE_NEW_LAPACK main_fast.cpp -framework Accelerate -o main_fast && ./main_fast
*/