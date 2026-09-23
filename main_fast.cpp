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
#include <deque>
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
    vector<float> sig;
};
class lupus{
    public:
        int n, d;
        vector<unordered_map<int, edge>> adj;
        vector<float> h; //nxdx1
        vector<float> force; //nxdx1
        vector<float> pred; //nxdx1
        vector<float> chl_err; //nxdx1
        vector<float> u; //nxdx1
        vector<float> v; //nxdx1
        vector<float> prec; //nxdx1
        vector<float> scaled_chl_err_p; //dx1
        vector<float> par_change; //dx1
        vector<float> chl_err_p; //nxdx1
        vector<bool> fixed; //nx1
        vector<int> deg;
        bool learn_prec=true;
        int tick=0;
        float slow_learn, fast_learn, eps, tanh_mag;
        void add_edge(int x1, int y1){
            //old randeye term 0.05f/sqrtf(d)
            adj[x1][y1]=edge{randeye(d,0.0f),vector<float>(d,0.0f)};
        }
        void make_sparse(int dg){
            uniform_int_distribution<int> disti(0,n-1);
            for (int i=0;i<n;i++){
                for (int j=0;j<dg;j++) add_edge(i, disti(rng));
            }
        }
        void reset(){
            tick=0;
            adj.assign(n, unordered_map<int, edge>{});
            //h.assign(n*d, 0.0f);
            h=randvec(n*d, 1.0f/sqrtf(d));
            force.assign(n*d, 0.0f);
            pred.assign(n*d, 0.0f);
            chl_err.assign(n*d, 0.0f);
            scaled_chl_err_p.assign(d, 0.0f);
            par_change.assign(d, 0.0f);
            u.assign(n*d, 0.0f);
            v.assign(n*d, 0.0f);
            prec.assign(n*d, 0.0f);
            chl_err_p.assign(n*d, 0.0f);
            fixed.assign(n, false); fixed[0]=true; fixed[1]=true; // fixed[3]=true;
            deg.assign(n,0);
            make_sparse(4);
            for (int i=0;i<n;i++){
                for (auto[j,_]:adj[i]){
                    deg[i]++; deg[j]++;
                }
            }
        }
        lupus(float un, float ud, float sl, float fl, float e, float tm){
            n=un; d=ud;
            slow_learn=sl; fast_learn=fl; eps=e; tanh_mag=tm;
            reset();
        }
        void forward(){
            fill(all(force), 0.0f);
            fill(all(pred), 0.0f);
            float err=0.0f;
            float wmg=0.0f;
            for (int i=0;i<n;i++) for (int j=0;j<d;j++) prec [i*d+j]=1.0f; //prec[i*d+j]=1.0f/(max(0.0f, u[i*d+j]-v[i*d+j]*v[i*d+j])+eps);
            for (int par=0;par<n;par++){
                for (auto& [i, e]:adj[par]){
                    auto& [w, sig]=e;
                    wmg+=mag(w, 0, d*d);
                    matvec(w, h, sig, d, d, 0, par);
                    for (int j=0;j<d;j++) {
                        sig[j]=tanh_mag*tanhf(sig[j]/tanh_mag);
                        pred[i*d+j]+=sig[j];
                    }
                }
            }
            for (int i=0;i<n;i++) {
                for (int j=0;j<d;j++){
                    chl_err[i*d+j]=pred[i*d+j]-h[i*d+j];
                    err+=0.5f*chl_err[i*d+j]*chl_err[i*d+j];
                    chl_err_p[i*d+j]=chl_err[i*d+j]*prec[i*d+j];
                    force[i*d+j]+=chl_err_p[i*d+j];
                    if (learn_prec){
                        u[i*d+j]+=dt*(chl_err[i*d+j]*chl_err[i*d+j]-u[i*d+j]);
                        v[i*d+j]+=dt*(chl_err[i*d+j]-v[i*d+j]);
                    }
                }
            }
            for (int par=0;par<n;par++){
                for (auto& [i, e]:adj[par]){
                    auto& [w, sig]=e;
                    for (int j=0;j<d;j++) {
                        scaled_chl_err_p[j]=chl_err_p[i*d+j]*(1-(sig[j]*sig[j])/(tanh_mag*tanh_mag));
                    }
                    matvec_transpose(w, scaled_chl_err_p, par_change, d, d, 0, 0);
                    for (int j=0;j<d;j++) force[par*d+j]-=par_change[j];
                    delta_rule(w, h, scaled_chl_err_p, slow_learn, d, d, par);
                }
            }
            // if (tick%1000==0){
            //     cout<<"err: "<<err<<"\nmag: "<<wmg<<'\n';
            //     cout<<"h_mag: "<<mag(h, 0, n*d)<<'\n';
            // }
            for (int i=0;i<n;i++){
                // for (int j=0;j<d;j++) force[i*d+j]/=max(1, deg[i]); 
                if (!fixed[i]) for (int j=0;j<d;j++) h[i*d+j]+=dt*fast_learn*force[i*d+j];
            }
            tick++;
        }
};
void train_sample(lupus& s, int tks, vector<float> ipt, vector<float> opt){
    // fill(all(s.h), 0.0f);
    // fill(all(s.u), 0.0f);
    // fill(all(s.v), 0.0f);
    for (int i=0;i<tks;i++) {
        for (int j=0;j<ipt.size();j++) {
            s.h[j]=ipt[j];
        }
        for (int j=0;j<opt.size();j++){
            s.h[s.d+j]=opt[j];
        }
        s.forward();
    }
}
vector<float> run_sample(lupus& s, int tks, vector<float> ipt){
    // fill(all(s.h), 0.0f);
    // s.learn_prec=false;
    float osl=s.slow_learn;
    s.fixed[1]=false;
    s.slow_learn=0.0f;
    for (int i=0;i<tks;i++){
        for (int j=0;j<s.d;j++) {
            s.h[j]=ipt[j];
        }
        s.forward();
    }
    vector<float> ret(s.d, 0.0f);
    for (int j=0;j<s.d;j++) {
        ret[j]=s.h[s.d+j];
    }
    s.slow_learn=osl;
    s.fixed[1]=true;
    s.learn_prec=true;
    return ret;
}
float f(float x){
    return sinf(x);
}
int main(){
    lupus sextus(25, 16, 0.01f, 1.0f, 0.2f, 1.0f);
    uniform_real_distribution<float> distf(-3.0f, 3.0f);
    for (int i=0;i<500;i++){
        cout<<"[train] "<<i<<'\n';
        float x=distf(rng); float y=f(x);
        train_sample(sextus, 5000, {x}, {y});
    }
    string fret="";
    fret+='{';
    int tot=100;
    for (int i=0;i<tot;i++){
        cout<<"[gen] "<<i<<'\n';
        float x=distf(rng);
        fret+="("+to_string(x)+", "+to_string(run_sample(sextus, 5000, {x})[0])+')';
        if (i<tot-1) fret+=", ";
    }
    fret+='}';
    cout<<fret<<'\n';
    return 0;
}
/*
clang++ -std=c++23 -O3 -Wall -DACCELERATE_NEW_LAPACK main_fast.cpp -framework Accelerate -o main_fast && ./main_fast
*/