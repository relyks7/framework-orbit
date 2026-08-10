#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <thread>
#include <iomanip>
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
struct edge{
    vector<float> w;
    vector<float> u; 
    vector<float> v;
};
class lupus{
    public:
        int n, d;
        vector<unordered_map<int, edge>> radj;
        vector<float> h; //nxdx1
        vector<float> dh; //nxdx1
        vector<float> force; //nxdx1
        vector<float> received_signal; //dx1
        vector<float> chl_err; //dx1
        vector<float> par_change; //dx1
        vector<float> precision; //dx1
        vector<float> chl_err_p; //dx1
        vector<float> err_p; //dx1
        vector<float> prior; //dx1
        vector<bool> input; //nx1
        float slow_learn, fast_learn, error_learn, eps;
        void reset(){
            radj.assign(n, unordered_map<int, edge>{});
            radj[0][1]=edge{randvec(d*d, 1.0f/sqrtf(d)), vector<float>(d, 0.0f), vector<float>(d, 0.0f)};
            radj[2][1]=edge{randvec(d*d, 1.0f/sqrtf(d)), vector<float>(d, 0.0f), vector<float>(d, 0.0f)};
            h.assign(n*d, 0.0f); h=randvec(n*d, 1.0f/sqrtf(d));
            dh.assign(n*d, 0.0f);
            force.assign(n*d, 0.0f);
            received_signal.assign(d, 0.0f);
            chl_err.assign(d, 0.0f);
            par_change.assign(d, 0.0f);
            precision.assign(d, 0.0f);
            chl_err_p.assign(d, 0.0f);
            err_p.assign(d, 0.0f);
            prior.assign(d, 0.0f);
            input.assign(n, false); input[0]=true;
        }
        lupus(float un, float ud, float sl, float fl, float el, float e){
            n=un; d=ud;
            slow_learn=sl; fast_learn=fl; error_learn=el; eps=e;
            reset();
        }
        void forward(){
            fill(all(force), 0.0f);
            for (int i=0;i<n;i++){
                fill(all(err_p), 0.0f);
                for (auto& [par, e]:radj[i]){
                    auto& [w, u, v]=e;
                    for (int j=0;j<d;j++) precision[j]=1.0f/(max(0.0f, u[j]-v[j]*v[j])+eps);
                    // for (int j=0;j<d;j++) precision[j]=1.0f;
                    matvec(w, h, received_signal, d, d, 0, par);
                    for (int j=0;j<d;j++) {
                        chl_err[j]=received_signal[j]-h[i*d+j];
                        chl_err_p[j]=chl_err[j]*precision[j];
                        err_p[j]+=chl_err_p[j];
                        u[j]+=dt*(chl_err[j]*chl_err[j]-u[j]);
                        v[j]+=dt*(chl_err[j]-v[j]);
                    }
                    matvec_transpose(w, dh, par_change, d, d, 0, i);
                    for (int j=0;j<d;j++) force[par*d+j]-=par_change[j];
                    delta_rule(w, h, chl_err_p, slow_learn, d, d, par);
                }
                for (int j=0;j<d;j++) force[i*d+j]+=err_p[j];
            }
            for (int i=0;i<n;i++){
                if (input[i]) for (int j=0;j<d;j++) force[i*d+j]+=prior[j];
                for (int j=0;j<d;j++) dh[i*d+j]+=dt*error_learn*(force[i*d+j]-dh[i*d+j]);
                if (!input[i]) for (int j=0;j<d;j++) h[i*d+j]+=dt*fast_learn*dh[i*d+j];
            }
        }
        vector<float> step(vector<float> ipt, vector<float> expected){
            prior=expected;
            vector<float> ret(d,0.0f);
            for (int i=0;i<d;i++) ret[i]-=h[2*d+i];
            for (int i=0;i<d;i++) h[0*d+i]=ipt[i];
            forward();
            for (int i=0;i<d;i++) ret[i]+=h[2*d+i];
            return ret;
        }
};
vector<float> flrs{15.0f, 20.0f};
vector<float> slrs{0.01f, 0.02f, 0.03f};
vector<float> elrs{2.0f, 4.0f, 7.0f, 10.0f, 15.0f};
vector<lupus> sextus_base{};
vector<int> total_success(100,0);
vector<pair<float, float>> trial{{1.0f,0.0f},{0.0f,1.0f},{-1.0f,0.0f},{0.0f,-1.0f},{0.8f,0.8f},{-0.8f,0.8f},{-0.8f,-0.8f},{0.8f,-0.8f},{0.15f,0.65f},{-0.7f,0.2f},{0.55f,-0.1f},{-0.25f,-0.9f},{0.9f,0.35f},{-0.4f,0.6f},{0.3f,-0.2f},{0.75f,0.1f},{0.3f,-0.2f},{-0.6f,-0.4f},{0.3f,-0.2f},{0.05f,0.05f},{-0.05f,0.05f},{0.05f,-0.05f},{-0.05f,-0.05f},{0.95f,-0.95f},{-0.95f,0.95f},{0.0f,0.0f}};
vector<float> j=randvec(4, 1.0f/sqrtf(2.0f));
int bestsuccess=-1;
float bestfl;
float bestsl;
float bestel;
int main(){
    cout<<"using j: \n"<<j[0]<<' '<<j[1]<<'\n'<<j[2]<<' '<<j[3]<<'\n';
    for (int i=0;i<100;i++) sextus_base.push_back(lupus(3, 4, 0.0f, 0.0f, 0.0f, 1.0f));
    for (auto flr:flrs){
        for (auto slr:slrs){
            for (auto elr:elrs){
                int success=0.0f;
                for (int _=0;_<100;_++){
                    lupus sextus=sextus_base[_]; sextus.fast_learn=flr; sextus.slow_learn=slr; sextus.error_learn=elr;
                    float curx=0.0f;
                    float cury=0.0f;
                    for (int i=0;i<100000;i++){
                        vector<float> mv=sextus.step({curx, cury, 0.0f, 0.0f}, {1.0f-curx, 2.0f-cury, 0.0f, 0.0f});
                        curx+=j[0]*mv[0]+j[1]*mv[1]; cury+=j[2]*mv[0]+j[3]*mv[1];
                    }
                    if (abs(curx-1.0f)<0.01f && abs(cury-2.0f)<0.01f) {
                        success++;
                        total_success[_]++;
                    }
                }
                cout<<"fast learn: "<<flr<<"\nslow learn: "<<slr<<"\nerror learn: "<<elr<<"\nsuccess rate: "<<success<<"/100\n";
                if (success>bestsuccess){
                    bestsuccess=success; bestfl=flr; bestsl=slr; bestel=elr;
                }
            }
        }
    }
    for (int i=0;i<100;i++){
        cout<<"initialization "<<i<<" success: "<<total_success[i]<<'\n';
    }
    while (true){
        int i; cin>>i;
        lupus sextus=sextus_base[i];
        sextus.fast_learn=bestfl;
        sextus.slow_learn=bestsl;
        sextus.error_learn=bestel;
        float curx=0.0f;
        float cury=0.0f;
        cout<<"trial:\n";
        for (auto [goalx, goaly]:trial){
            int timer=0;
            bool converged=false;
            for (int i=0;i<100000;i++){
                vector<float> mv=sextus.step({curx, cury, 0.0f, 0.0f}, {goalx-curx, goaly-cury, 0.0f, 0.0f});
                curx+=j[0]*mv[0]+j[1]*mv[1]; cury+=j[2]*mv[0]+j[3]*mv[1];
                //if (i%1000) cout<<curx<<' '<<cury<<'\n';
                if (abs(curx-goalx)<0.01f && abs(cury-goaly)<0.01f) {
                    timer++;
                } else timer=0;
                if (timer>200){
                    cout<<"tick "<<i<<", converged to ("<<goalx<<", "<<goaly<<")\n";
                    converged=true;
                    break;
                }
            }
            if (!converged) {
                cout<<"did not converge to ("<<goalx<<", "<<goaly<<")\n";
                break;
            }
        }
    }
    return 0;
}
/*
clang++ -std=c++23 -O3 -Wall -DACCELERATE_NEW_LAPACK main_fast.cpp -framework Accelerate -o main_fast && ./main_fast
*/