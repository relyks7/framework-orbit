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
typedef pair<vector<float>, vector<float>> pvf;
using ll=long long;
int INF=0x3f3f3f3f;
float SENTINEL=-MAXFLOAT;
thread_local mt19937 rng(hash<thread::id>{}(this_thread::get_id())^random_device{}());
float gaussian_noise(float mean, float stddev){
    normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}
thread_local uniform_real_distribution<float> disf(-5.0f, 5.0f);
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
            a[i*m+j]+=-lr*err[i]*x[idx*m+j];
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
    vector<float> b;
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
        float slow_learn, fast_learn, bias_learn, eps, tanh_mag;
        void add_edge(int x1, int y1){
            //old randeye term 0.05f/sqrtf(d)
            adj[x1][y1]=edge{randeye(d,0.0f/sqrtf(d)),vector<float>(d,0.0f),vector<float>(d,0.0f)};
        }
        void bi(int x1, int y1){
            add_edge(x1, y1); add_edge(y1, x1);
        }
        void make_graph(){
            //evil squared square polygonal electric network
            //why does this work
            //o24-195a
            bi(0, 3);
            bi(0, 2);
            bi(0, 1);
            bi(1, 2);
            bi(1, 5);
            bi(2, 3);
            bi(2, 4);
            bi(2, 5);
            bi(3, 8);
            bi(3, 6);
            bi(3, 4);
            bi(4, 6);
            bi(4, 7);
            bi(5, 7);
            bi(5, 9);
            bi(6, 10);
            bi(7, 11);
            bi(8, 12);
            bi(8, 10);
            bi(9, 11);
            bi(9, 12);
            bi(10, 12);
            bi(10, 11);
            bi(11, 12);
            bi(0, 12);
        }
        void reset(){
            tick=0;
            h.assign(n*d, 0.0f);
            force.assign(n*d, 0.0f);
            pred.assign(n*d, 0.0f);
            chl_err.assign(n*d, 0.0f);
            scaled_chl_err_p.assign(d, 0.0f);
            par_change.assign(d, 0.0f);
            u.assign(n*d, 0.0f);
            v.assign(n*d, 0.0f);
            prec.assign(n*d, 0.0f);
            chl_err_p.assign(n*d, 0.0f);
            fixed.assign(n, false);
            fixed[0]=true; //s_t
            fixed[1]=true; //s_t+1
            fixed[2]=true; //err
            fixed[n-1]=true; //act
            for (int par=0;par<n;par++){
                for (auto& [i, e]:adj[par]){
                    e=edge{randeye(d,0.0f/sqrtf(d)),vector<float>(d,0.0f),vector<float>(d,0.0f)};
                }
            }
        }
        lupus(int un, int ud, float fl, float e, float tm){
            n=un; d=ud;
            fast_learn=fl;
            eps=e;
            tanh_mag=tm;
            adj.assign(n, unordered_map<int, edge>{});
            make_graph();
            deg.assign(n,0);
            for (int i=0;i<n;i++){
                for (auto[j,_]:adj[i]){
                    deg[i]++; deg[j]++;
                }
            }
            reset();
        }
        void forward(){
            fill(all(force), 0.0f);
            fill(all(pred), 0.0f);
            float err=0.0f;
            float wmg=0.0f;
            for (int i=0;i<n;i++) for (int j=0;j<d;j++) prec[i*d+j]=1.0f; //prec[i*d+j]=1.0f/(max(0.0f, u[i*d+j]-v[i*d+j]*v[i*d+j])+eps);
            for (int par=0;par<n;par++){
                for (auto& [i, e]:adj[par]){
                    auto& [w, b, sig]=e;
                    wmg+=mag(w, 0, d*d);
                    matvec(w, h, sig, d, d, 0, par);
                    for (int j=0;j<d;j++) sig[j]+=b[j];
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
                    auto& [w, b, sig]=e;
                    for (int j=0;j<d;j++) {
                        scaled_chl_err_p[j]=chl_err_p[i*d+j]*(1-(sig[j]*sig[j])/(tanh_mag*tanh_mag));
                    }
                    matvec_transpose(w, scaled_chl_err_p, par_change, d, d, 0, 0);
                    for (int j=0;j<d;j++) force[par*d+j]-=par_change[j];
                    delta_rule(w, h, scaled_chl_err_p, slow_learn, d, d, par);
                    for (int j=0;j<d;j++) b[j]+=-bias_learn*scaled_chl_err_p[j];
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
        void train(int tks, vector<float> ipt0, vector<float> ipt1, vector<float> ipt2, vector<float> opt, float lr){
            fill(all(h), 0.0f);
            fill(all(u), 0.0f);
            fill(all(v), 0.0f);
            slow_learn=0.0f;
            bias_learn=0.0f;
            for (int i=0;i<tks;i++) {
                for (int j=0;j<ipt0.size();j++) {
                    h[j]=ipt0[j];
                }
                for (int j=0;j<ipt1.size();j++) {
                    h[1*d+j]=ipt1[1*d+j];
                }
                for (int j=0;j<ipt2.size();j++) {
                    h[2*d+j]=ipt2[2*d+j];
                }
                for (int j=0;j<opt.size();j++){
                    h[(n-1)*d+j]=opt[j];
                }
                forward();
            }
            slow_learn=lr;
            bias_learn=lr;
            for (int j=0;j<ipt0.size();j++) {
                h[j]=ipt0[j];
            }
            for (int j=0;j<ipt1.size();j++) {
                h[1*d+j]=ipt1[1*d+j];
            }
            for (int j=0;j<ipt2.size();j++) {
                h[2*d+j]=ipt2[2*d+j];
            }
            for (int j=0;j<opt.size();j++){
                h[(n-1)*d+j]=opt[j];
            }
            forward();
        }
        vector<float> generate(int tks, vector<float> ipt0, vector<float> ipt1, vector<float> ipt2){
            fill(all(h), 0.0f);
            fill(all(u), 0.0f);
            fill(all(v), 0.0f);
            float osl=slow_learn;
            float obl=bias_learn;
            slow_learn=0.0f;
            bias_learn=0.0f;
            fixed[n-1]=false;
            for (int i=0;i<tks;i++) {
                for (int j=0;j<ipt0.size();j++) {
                    h[j]=ipt0[j];
                }
                for (int j=0;j<ipt1.size();j++) {
                    h[1*d+j]=ipt1[1*d+j];
                }
                for (int j=0;j<ipt2.size();j++) {
                    h[2*d+j]=ipt2[2*d+j];
                }
                forward();
                
            }
            vector<float> ret(d, 0.0f);
            for (int j=0;j<d;j++) {
                ret[j]=h[(n-1)*d+j];
            }
            slow_learn=osl;
            bias_learn=obl;
            fixed[n-1]=true;
            return ret;
        }
};
int traincnt=1000;
int testcnt=1000;
int main(){
    lupus sextus(13, 1, 10.0f, 1.0f, 1.5f);
    for (int _=1;_<=traincnt;_++){
        float x=disf(rng), y=sinf(x);
        sextus.train(5000, {x}, {y}, 0.3f);
        if (_%50==0) cout<<"train "<<_<<'\n';
    }
    float mse=0.0f;
    for (int _=1;_<=testcnt;_++){
        float x=disf(rng), y=sextus.generate(1000, {x})[0];
        float cy=sinf(x);
        mse+=(y-cy)*(y-cy)/testcnt;
        if (_%50==0) cout<<"test "<<_<<'\n';
    }
    cout<<mse<<'\n';
}
/*
clang++ -std=c++23 -O3 -Wall -DACCELERATE_NEW_LAPACK main_fast.cpp -framework Accelerate -o main_fast && ./main_fast
*/