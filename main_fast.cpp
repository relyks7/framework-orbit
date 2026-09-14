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
class lupus{
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
        vector<int> deg;
        int tick=0;
        float slow_learn, fast_learn, eps, tanh_mag;
        void add_edge(int x1, int y1){
            adj[x1][y1]=edge{randeye(d,0.05f/sqrtf(d)),vector<float>(d,0.0f),vector<float>(d,0.0f)};
        }
        void reset(){
            tick=0;
            adj.assign(n, unordered_map<int, edge>{});
            h.assign(n*d, 0.0f); // h=randvec(n*d, 1.0f/sqrtf(d));
            force.assign(n*d, 0.0f);
            received_signal.assign(d, 0.0f);
            chl_err.assign(d, 0.0f);
            scaled_chl_err_p.assign(d, 0.0f);
            par_change.assign(d, 0.0f);
            precision.assign(d, 0.0f);
            chl_err_p.assign(d, 0.0f);
            fixed.assign(n, false); // fixed[0]=true; fixed[2]=true; fixed[3]=true;
            deg.assign(n,0);
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
                }
            }
            for (int i=0;i<n;i++){
                for (int j=0;j<d;j++) force[i*d+j]/=max(1, deg[i]);
                if (!fixed[i]) for (int j=0;j<d;j++) h[i*d+j]+=dt*fast_learn*force[i*d+j];
            }
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
int main(){
    lupus sextus(5, 4, 0.01f, 5.0f, 1.0f, 8.0f);
    return 0;
}
/*
clang++ -std=c++23 -O3 -Wall -DACCELERATE_NEW_LAPACK main_fast.cpp -framework Accelerate -o main_fast && ./main_fast
*/